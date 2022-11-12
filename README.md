# transformer-attention-visualizer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A lightweight toolkit for inspecting scaled dot-product attention patterns, token attribution maps, and attention head entropy in Transformer architectures.

---

## Attention Formulation

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Head Entropy:

$$
H(\text{Head}) = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^N A_{ij} \log_2 A_{ij}
$$

---

## Quickstart

```bash
git clone https://github.com/GuptaSoujanya/transformer-attention-visualizer.git
cd transformer-attention-visualizer
pip install -e .
python -m attention_viz.cli
```

---

## License

MIT License (c) 2024 Soujanya Gupta.
