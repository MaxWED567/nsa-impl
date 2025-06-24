# NSA: Native Sparse Attention

A PyTorch+Triton+FlexAttention implementation of Neighborhood-Selective Attention (NSA) that combines compression, selection, and sliding window attention mechanisms.

For a deep dive into sparse attention mechanisms and the design of this kernel, check out our blog post: [Sparsity is Cool?](https://www.tilderesearch.com/blog/sparse-attn)

## Installation

```bash
# Using uv (recommended)
uv sync
```

## Usage

```python
import torch
from nsa import nsa_func
# Run NSA
output = nsa_func(
    q, k, v,
    g_cmp=g_cmp,
    g_slc=g_slc,
    g_swa=g_swa,
    block_counts=16,
    block_size=16,  # Must be >= 16
    window_size=32,
    scale=None  # Defaults to 1/sqrt(D)
)
```

## Features

- Supports toggling between one-pass (atomic) and two-pass backward variants for selection attention
- GQA (Grouped Query Attention) compatible
- Efficient Triton kernels for high throughput

## Acknowledgments

This implementation uses components from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), specifically the parallel NSA implementation for the two-pass variant. We thank the FLA team for their excellent work on efficient attention mechanisms.

The kernel has been implemented following the Native Sparse Attention paper by DeepSeek: [arXiv:2502.11089](https://arxiv.org/abs/2502.11089).

