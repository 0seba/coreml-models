# CoreML Models

This repository contains some Transformers, layers and other operations in CoreML.

### On ANE-compatible flexible input

- Linear by itself can work with flexible input on ANE, but when using them in a more complex graph (like a transformer arch), it stops running on ANE, convolutions are more suitable for ANE, which also means using channels first.
- Use channels first, convolutions are more suitable for ANE.
- In order to build attention mask and rope embeddings for flexible inputs I had to use `fill_like` and `cumsum` opsets, to build indices used to gather from the complete arrays. Using any operation that received a shape as input, such as `fill_like`, `range_1d`, `slice_by_index`, `slice_by_size`, caused the whole model to not run on ANE, for example using something in the like of `mb.slice_by_size(..., size=mb.shape(some_flexible_shape_tensor))` did not work.
- The only attention implementation that I managed to get working on ANE with flexible shape is the following:
    - Receive QKV: (batch size, qkv dimension, flexible sequence length)
    - Split into q, k, v individual heads: QKV->split into (num q heads + num k heads + num v heads) tensors with shape (batch size, head dimension, flexible sequence length)
    - Iterate all heads, perform normalization and rope transform on each head individually, perform attention on each individual head, and concatenate all of them at the end
- Other attempts that tried to first reshape into big tensors that contain all heads, and normalize and rope transformer many heads at the time did not work. Multiplications that had shape (batch size, 1, head dim, sequence length) or (batch size, several heads, head dim, sequence length) couldn't be run on ANE.
- The working implementation has the issue that since normalizations, rope transformers and attentions are computed on a per head basis, the resulting graph has a lot of operations, thus it is slower to load.