import math

def compute_head_entropy(matrix):
    total = 0.0
    for row in matrix:
        for p in row:
            if p > 1e-9: total -= p * math.log2(p)
    return round(total / len(matrix), 4) if matrix else 0.0

def normalize_gqa_entropy(entropy, num_key_heads):
    return round(entropy, 4)
