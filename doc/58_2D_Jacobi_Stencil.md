# 2D Jacobi Stencil

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/2d-jacobi-stencil)

Given a 2D grid of 32-bit floating point values, apply one iteration of the 5-point Jacobi stencil: each interior cell of the output is set to the average of its four cardinal neighbors (top, bottom, left, right) from the input grid. Boundary cells (first/last row and column) are copied unchanged from the input to the output.

![](../assets/58_2D_Jacobi_Stencil/image_01.svg)

### Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in `output`
- Read exclusively from `input` and write exclusively to `output` (do not update `input`)

### Example:

Input ($4 \times 4$): $$ \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \\ 5.0 & 6.0 & 7.0 & 8.0 \\ 9.0 & 10.0 & 11.0 & 12.0 \\ 13.0 & 14.0 & 15.0 & 16.0 \end{bmatrix} $$ Output ($4 \times 4$): $$ \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \\ 5.0 & 6.0 & 7.0 & 8.0 \\ 9.0 & 10.0 & 11.0 & 12.0 \\ 13.0 & 14.0 & 15.0 & 16.0 \end{bmatrix} $$ Interior cell $(1,1)$: $0.25 \times (\text{input}[0,1] + \text{input}[2,1] + \text{input}[1,0] + \text{input}[1,2])$ $= 0.25 \times (2.0 + 10.0 + 5.0 + 7.0) = 6.0$  
Interior cell $(1,2)$: $0.25 \times (\text{input}[0,2] + \text{input}[2,2] + \text{input}[1,1] + \text{input}[1,3])$ $= 0.25 \times (3.0 + 11.0 + 6.0 + 8.0) = 7.0$  
Interior cell $(2,1)$: $0.25 \times (\text{input}[1,1] + \text{input}[3,1] + \text{input}[2,0] + \text{input}[2,2])$ $= 0.25 \times (6.0 + 14.0 + 9.0 + 11.0) = 10.0$  
Interior cell $(2,2)$: $0.25 \times (\text{input}[1,2] + \text{input}[3,2] + \text{input}[2,1] + \text{input}[2,3])$ $= 0.25 \times (7.0 + 15.0 + 10.0 + 12.0) = 11.0$

### Constraints

- 1 ≤ `rows`, `cols` ≤ 16,384
- Input values are in the range \[-100, 100\]
- All values are 32-bit floats
- Performance is measured with `rows` = 8,192, `cols` = 8,192

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/2d-jacobi-stencil)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
