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

import ast, math, csv, os
from typing import List, Tuple, Dict, Optional

def load_events(path: str) -> List[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
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
    # Build normalized points
    norm = []
    for toks, L in eval_curve:
        z = (L - L0) / (logV - L0)
        norm.append((toks, z))

    # scan for first segment crossing z_target
    for (t1, z1), (t2, z2) in zip(norm, norm[1:]):
        if (z1 <= z_target <= z2) or (z2 <= z_target <= z1):
            # linear interpolation
            if z2 == z1:
                return float(t2)  # flat segment; return the right edge
            alpha = (z_target - z1) / (z2 - z1)
            return float(t1 + alpha * (t2 - t1))
    return None

def initial_slope(
    eval_curve: List[Tuple[int, float]],
    L0: float,
    logV: float
) -> Optional[float]:
    """
    Finite-difference slope (Δz / Δtokens) using the first two eval points.
    Smaller magnitude => more robust (slower drift toward uniform).
    """
    if len(eval_curve) < 2 or logV == L0:
        return None
    (t1, L1), (t2, L2) = eval_curve[0], eval_curve[1]
    z1 = (L1 - L0) / (logV - L0)
    z2 = (L2 - L0) / (logV - L0)
    dt = t2 - t1
    if dt <= 0:
        return None
    return (z2 - z1) / dt

def compute_metrics_for_run(
    path: str,
    vocab_size: int,
    tokens_per_step: int
) -> Dict[str, Optional[float]]:
    events = load_events(path)
    eval_points = extract_eval_points(events, tokens_per_step)
    assert eval_points, "No eval rows found in the log file"

    # Baseline clean eval loss L0: first eval point
    L0 = float(eval_points[0][1])
    logV = math.log(vocab_size) # this is the expected target loss (when the model learns a uniform distribution)

    t_half = interpolate_tokens_for_z_target(eval_points, L0, logV, 0.5)
    t_90 = interpolate_tokens_for_z_target(eval_points, L0, logV, 0.9)
    slope0 = initial_slope(eval_points, L0, logV)

    return {
        "tokens_at_z0.5": t_half,
        "tokens_at_z0.9": t_90,
        "initial_slope": slope0,
        "L0": L0,
        "logV": logV,
        "n_eval_points": len(eval_points)
    }

def main():
    results_dir = "results"
    args_list = [
        {
            "label": "Mistral-7B",
            "filename": "Mistral-7B-v0.3.jsonl",
            "vocab_size": 32768,
            "tokens_per_step": 128000
        },
        {
            "label": "Qwen2.5-7B",
            "filename": "Qwen2.5-7B.jsonl",
            "vocab_size": 151665,
            "tokens_per_step": 128000
        },
        {
            "label": "Qwen2.5-3B",
            "filename": "Qwen2.5-3B.jsonl",
            "vocab_size": 151665,
            "tokens_per_step": 128000
        }
    ]

    rows = []
    for args in args_list:
        label = args['label']
        path = os.path.join(results_dir, args['filename'])
        V = args['vocab_size']
        tokens_per_step = args['tokens_per_step']
        metrics = compute_metrics_for_run(path, V, tokens_per_step)
        rows.append({
            "label": label,
            "log_file": path,
            "vocab_size": V,
            "tokens_per_step": tokens_per_step,
            "L0": metrics["L0"],
            "logV": metrics["logV"],
            "tokens_at_z0.5": metrics["tokens_at_z0.5"],
            "tokens_at_z0.9": metrics["tokens_at_z0.9"],
            "initial_slope_z_per_token": metrics["initial_slope"],
            "n_eval_points": metrics["n_eval_points"]
        })

    # Write CSV
    with open("metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Pretty print
    print(f"\nWrote: metrics.csv\n")
    for r in rows:
        print(
            f"{r['label']}: "
            f"L0={r['L0']:.4f} | logV={r['logV']:.4f} | "
            f"half-life tokens={int(r['tokens_at_z0.5']):,} | "
            f"90%-to-uniform tokens={int(r['tokens_at_z0.9']):,} | "
            f"initial slope={r['initial_slope_z_per_token']}"
        )

if __name__ == "__main__":
    main()