"""
confusability_metrics.py

Parse HF Trainer-like JSON logs and compute confusability/robustness metrics:
- z(t) = (L_eval(t) - L0) / (log(V) - L0)
- Half-life tokens: tokens to reach z=0.5 (linear interpolation)
- 90%-to-uniform tokens: tokens to reach z=0.9
- Initial slope: (z2 - z1) / (tokens2 - tokens1) using the first two eval points

Assumptions:
- Log file has one JSON object per line (train logs have "loss", eval logs have "eval_loss").
- Prefer "step" or "global_step" field if present; otherwise we count train logs to infer step indices.
- You provide effective tokens per optimizer step (batch_size * grad_accum * seq_len).
- You provide vocab size V for each run (we use natural log).
"""

import ast, math
from typing import List, Tuple, Dict, Optional

def load_events(path: str) -> List[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            obj = ast.literal_eval(line.strip())
            events.append(obj)
    return events

def extract_eval_points(
    events: List[dict],
    tokens_per_step: int
) -> List[Tuple[int, float]]:
    """
    Returns a list of (tokens_seen, eval_loss), ordered by time.
    """
    eval_points = []
    step_counter = 0

    for ev in events:
        if "eval_loss" in ev:
            tokens_seen = int(tokens_per_step * step_counter)
            eval_points.append((tokens_seen, float(ev["eval_loss"])))
        elif "loss" in ev:
            step_counter += 1

    # Ensure sorted by tokens_seen
    assert eval_points == sorted(eval_points, key=lambda x: x[0])
    return eval_points

def eval_curve_to_z_curve(
    eval_curve: List[Tuple[int, float]],
    L0: float,
    logV: float
) -> List[Tuple[int, float]]:
    z_curve = []
    for toks, L in eval_curve:
        z = (L - L0) / (logV - L0)
        z_curve.append((toks, z))
    return z_curve

def interp_t(z_a: float, z_b: float, t_a: float, t_b: float, z: float) -> float:
    # Linear interpolation in z space
    if z_b == z_a:
        return t_b
    alpha = (z - z_a) / (z_b - z_a)
    return t_a + alpha * (t_b - t_a)

def interpolate_tokens_for_z_target(
    eval_curve: List[Tuple[int, float]],
    L0: float,
    logV: float,
    z_target: float
) -> Optional[float]:
    """
    Given (tokens, L_eval) points, return tokens where z reaches z_target using linear interpolation.
    z = (L - L0)/(logV - L0). Returns None if never reaches z_target.
    """
    # Build normalized points via helper
    z_curve = eval_curve_to_z_curve(eval_curve, L0, logV)

    # scan for first segment crossing z_target
    for (t1, z1), (t2, z2) in zip(z_curve, z_curve[1:]):
        if (z1 <= z_target <= z2) or (z2 <= z_target <= z1):
            return interp_t(z1, z2, t1, t2, z_target)
    return None

def fit_loss_slope_until_z(
    eval_curve: List[Tuple[int, float]],
    L0: float,
    logV: float,
    z_max: float = 0.3,
) -> Optional[float]:
    """
    Fit a least-squares line to (tokens, eval_loss) using only the initial
    contiguous eval points whose normalized z does not exceed z_max.

    z is defined as z = (L - L0) / (logV - L0).

    Returns d(loss)/d(token) (loss change per token). Returns None if fewer
    than 2 points are available or if variance in tokens is zero.
    """
    # Collect points until first z > z_max
    z_curve = eval_curve_to_z_curve(eval_curve, L0, logV)
    if not z_curve:
        return None
    
    used = [(t, z) for t, z in z_curve if z <= z_max]

    if len(used) < 2:
        return None

    # Least squares slope
    xs = [t for t, _ in used]
    ys = [l for _, l in used]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in used)
    den = sum((x - x_mean) ** 2 for x, _ in used)
    if den == 0:
        return None
    return num / den

def auc_tokens_until_z(
    eval_curve: List[Tuple[int, float]],
    L0: float,
    logV: float,
    z_max: float,
    include_origin: bool = True,
    normalize: bool = False,
) -> Optional[float]:
    """
    Compute the area under the curve of tokens as a function of z, from z=0 to
    z=z_max, using piecewise-linear interpolation between eval points.

    - Each eval point provides (tokens, loss). We convert loss -> z using
      z = (L - L0) / (logV - L0).
    - We then integrate t(z) dz up to z_max using trapezoids per segment.
    - If include_origin is True, we assume (z=0, tokens=0) as the start.

    Returns:
        - The AUC (float) in units of tokens*z. If normalize=True, returns the
          mean tokens over [0, z_max], i.e., AUC / z_max.
        - None if there are insufficient segments overlapping [0, z_max]
          (e.g., all z are above z_max or denominator is zero).
    """
    denom = (logV - L0)
    if denom == 0:
        return None

    # Build (tokens, z) points
    z_curve = eval_curve_to_z_curve(eval_curve, L0, logV)
    if include_origin:
        z_curve = [(0, 0.0)] + z_curve

    # Sort by z to integrate t(z) over z
    z_curve.sort(key=lambda p: p[1])

    # Integrate over overlapping parts with [0, z_max]
    auc = 0.0
    coverage = 0.0
    had_overlap = False

    for (t1, z1), (t2, z2) in zip(z_curve, z_curve[1:]):
        if z1 == z2:
            continue
        z_left = max(min(z1, z2), 0.0)
        z_right = min(max(z1, z2), z_max)
        if z_right <= z_left:
            continue

        had_overlap = True
        # Ensure (z1, t1) is the left point in z
        if z2 < z1:
            z1, z2 = z2, z1
            t1, t2 = t2, t1

        t_left = interp_t(z1, z2, t1, t2, z_left)
        t_right = interp_t(z1, z2, t1, t2, z_right)
        width = (z_right - z_left)
        auc += 0.5 * (t_left + t_right) * width
        coverage += width

    if not had_overlap:
        return None

    if normalize:
        # Normalize by full target range if include_origin, else by actual coverage
        denom_z = z_max if include_origin else (coverage if coverage > 0 else None)
        if not denom_z:
            return None
        return auc / denom_z

    return auc