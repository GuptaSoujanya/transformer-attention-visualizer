import argparse
from .core import compute_attention
from .entropy import compute_head_entropy

def main():
    q = [[1.0, 0.0], [0.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    attn = compute_attention(q, k)
    h = compute_head_entropy(attn)
    print("\n" + "="*50)
    print("       TRANSFORMER ATTENTION MATRIX")
    print("="*50)
    for row in attn:
        print("  " + "  ".join(f"{v:.4f}" for v in row))
    print(f"\n Head Entropy: {h} bits (Shannon)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
