# LoRA Linear

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/lora-linear)

Implement a LoRA (Low-Rank Adaptation) linear layer forward pass. Given an input matrix `x` of shape `batch × d_in`, a base weight matrix `W` of shape `d_out × d_in`, a LoRA down-projection matrix `A` of shape `rank × d_in`, and a LoRA up-projection matrix `B` of shape `d_out × rank`, compute `output = x × W`<sup>`T`</sup>` + lora_scale × (x × A`<sup>`T`</sup>`) × B`<sup>`T`</sup>. All tensors are `float32`.

![](../assets/69_LoRA_Linear/image_01.svg)

### Implementation Requirements

- Implement the `solve` function; do not change its signature.
- Do not use external libraries beyond those provided.
- Write the result into `output`.

### Examples

Example 1:

$$ x = \begin{bmatrix} 1 & 0 & -1 & 2 \\ 0 & 1 & 1 & -1 \end{bmatrix},\quad W = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix},\quad A = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix},\quad B = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} $$

With `lora_scale` = 0.5:

$$ \text{output} = x W^T + 0.5 \cdot (x A^T) B^T = \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 1 \end{bmatrix} + 0.5 \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 1.5 & 0 & -1 \\ 0 & 1.5 & 1 \end{bmatrix} $$

### Constraints

- 1 ≤ `batch` ≤ 1,024
- 1 ≤ `d_in`, `d_out` ≤ 8,192
- 1 ≤ `rank` ≤ 256; `rank` \< min(`d_in`, `d_out`)
- All tensors are `float32` on GPU.
- Performance is measured with `batch` = 256, `d_in` = 4,096, `d_out` = 4,096, `rank` = 64

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/lora-linear)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
