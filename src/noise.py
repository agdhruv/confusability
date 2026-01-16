"""
Noise strategy for label corruption experiments.
"""
import re
import random


def systematic_offset(answer: str, offset: int = 3) -> str:
    """Shift every number by a fixed offset (default +3)."""
    pattern = r'-?\d+\.?\d*'
    
    def shift_match(match):
        num_str = match.group()
        if '.' in num_str:
            num = float(num_str)
            shifted = num + int(offset)
            return str(shifted)
        num = int(num_str)
        return str(num + int(offset))
    try:
        return re.sub(pattern, shift_match, answer)
    except Exception as e:
        breakpoint()
        raise e


def number_perturbation(answer: str, rng: random.Random, scale: float = 0.3) -> str:
    """
    Perturb numbers in the answer by multiplying by (1 + noise) where noise ~ N(0, scale).
    """
    pattern = r'-?\d+\.?\d*'
    
    def perturb_match(match):
        num_str = match.group()
        try:
            num = float(num_str)
            if num == 0:
                noisy = rng.gauss(0, scale * 10)
            else:
                noisy = num * (1 + rng.gauss(0, scale))
            
            if '.' not in num_str:
                return str(int(round(noisy)))
            else:
                decimals = len(num_str.split('.')[1])
                return f"{noisy:.{decimals}f}"
        except ValueError:
            return num_str
    
    return re.sub(pattern, perturb_match, answer)
