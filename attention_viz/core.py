import math

def softmax(vec):
    if not vec: return []
    m = max(vec)
    e = [math.exp(x - m) for x in vec]
    s = sum(e)
    return [round(x/s, 4) for x in e]

def compute_attention(q_mat, k_mat):
    d_k = len(q_mat[0])
    scale = math.sqrt(d_k)
    weights = []
    for q in q_mat:
        row = [sum(a*b for a, b in zip(q, k)) / scale for k in k_mat]
        weights.append(softmax(row))
    return weights
