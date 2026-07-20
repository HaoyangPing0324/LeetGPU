# Group Normalization

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/group-normalization)

Implement Group Normalization for 4D activation tensors, the normalization layer used by Stable Diffusion U-Nets and many ResNet variants. Given an input tensor `X` of shape `(N, C, H, W)`, the channels are split into `G` contiguous groups of `C/G` channels each. For every `(batch, group)` pair, the mean and variance are computed over all `(C/G) × H × W` elements, the activations are normalized, then scaled and shifted by per-channel parameters `gamma` and `beta`.

For each batch index `n` and group index `g`, let \$ \mathcal{S}\_{n,g} = \\(n, c, h, w) : c \in \[g \cdot C/G,\\ (g+1) \cdot C/G)\\ \$. Group Normalization computes: $$ \begin{align} \mu\_{n,g} &= \frac{1}{\|\mathcal{S}\_{n,g}\|} \sum\_{(n,c,h,w) \in \mathcal{S}\_{n,g}} x\_{n,c,h,w} \\ \sigma\_{n,g}^2 &= \frac{1}{\|\mathcal{S}\_{n,g}\|} \sum\_{(n,c,h,w) \in \mathcal{S}\_{n,g}} (x\_{n,c,h,w} - \mu\_{n,g})^2 \\ \hat{x}\_{n,c,h,w} &= \frac{x\_{n,c,h,w} - \mu\_{n,g(c)}}{\sqrt{\sigma\_{n,g(c)}^2 + \epsilon}} \\ y\_{n,c,h,w} &= \gamma_c \\ \hat{x}\_{n,c,h,w} + \beta_c \end{align} $$ where \$ g(c) = \lfloor c \cdot G / C \rfloor \$ maps a channel to its group.

![](../assets/75_Group_Normalization/image_01.svg)

### Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `Y` tensor

### Example 1:

    Input:  N=1, C=4, H=2, W=2, G=2, eps=1e-5
            X[0,0] = [[1, 1], [1, 1]]
            X[0,1] = [[3, 3], [3, 3]]
            X[0,2] = [[2, 2], [2, 2]]
            X[0,3] = [[6, 6], [6, 6]]
            gamma = [1, 1, 1, 1]
            beta  = [0, 0, 0, 0]
    Output: Y[0,0] = [[-1, -1], [-1, -1]]
            Y[0,1] = [[ 1,  1], [ 1,  1]]
            Y[0,2] = [[-1, -1], [-1, -1]]
            Y[0,3] = [[ 1,  1], [ 1,  1]]
    Note:   Group 0 = channels {0, 1}: mean = 2, var = 1, std = 1
            Group 1 = channels {2, 3}: mean = 4, var = 4, std = 2

### Example 2:

    Input:  N=1, C=2, H=1, W=2, G=2, eps=1e-5
            X[0,0] = [[1, 3]]
            X[0,1] = [[2, 6]]
            gamma = [2, 1]
            beta  = [0, 0]
    Output: Y[0,0] = [[-2,  2]]
            Y[0,1] = [[-1,  1]]
    Note:   G=C, so each channel is its own group (Instance Norm).
            Channel 0: mean=2, var=1, std=1
            Channel 1: mean=4, var=4, std=2

### Constraints

- 1 ≤ `N` ≤ 32
- 1 ≤ `C` ≤ 1,024 and `C` is divisible by `G`
- 1 ≤ `G` ≤ `C`
- 1 ≤ `H`, `W` ≤ 128
- `eps` = 1e-5
- -100.0 ≤ input values ≤ 100.0
- 0.1 ≤ `gamma` values ≤ 10.0
- -10.0 ≤ `beta` values ≤ 10.0
- Performance is measured with `N` = 8, `C` = 512, `H` = 64, `W` = 64, `G` = 32

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/group-normalization)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
