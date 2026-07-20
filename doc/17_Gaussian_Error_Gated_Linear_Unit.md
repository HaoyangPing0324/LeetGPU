# Gaussian Error Gated Linear Unit

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Easy](../assets/common/difficulty_easy.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/gaussian-error-gated-linear-unit)

Implement the Gaussian Error Gated Linear Unit (GEGLU) activation function forward pass for 1D input vectors. Given an input tensor of shape \[N\] where N is the number of elements, compute the output using the elementwise formula. The input and output tensor must be of type `float32`.

GEGLU is defined as:

1.  Split input $x$ into two halves: $x_1$ and $x_2$
2.  Compute GELU on the second half: $$ \text{GELU}(x_2) = \frac{1}{2} x_2 \left(1 + \text{erf}\left(\frac{x_2}{\sqrt{2}}\right)\right) $$
3.  Compute the GEGLU output: $$ \text{GEGLU}(x_1, x_2) = x_1 \cdot \text{GELU}(x_2) $$

### Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` tensor

### Example 1:

    Input:  [1.0, 1.0]  (N=2)
    Output: [0.8413447]

### Example 2:

    Input:  [2.0, -1.0, 1.0, 0.5]  (N=4)
    Output: [1.6826895, -0.3457312]

### Constraints

- 1 ≤ `N` ≤ 1,000,000
- N is an even number
- -100.0 ≤ input values ≤ 100.0
- Performance is measured with `N` = 1,000,000

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/gaussian-error-gated-linear-unit)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
