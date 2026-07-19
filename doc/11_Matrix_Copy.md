# Matrix Copy

## Problem Description

![Difficulty: Easy](../assets/common/difficulty_easy.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/matrix-copy)

Implement a program that copies an $N \times N$ matrix of 32-bit floating point numbers from input array $A$ to output array $B$ on the GPU. The program should perform a direct element-wise copy so that $B_{i,j} = A_{i,j}$ for all valid indices.

### Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in matrix `B`

### Example 1:

    Input:  A = [[1.0, 2.0],
                 [3.0, 4.0]]
    Output: B = [[1.0, 2.0],
                 [3.0, 4.0]]

### Example 2:

    Input:  A = [[5.5, 6.6, 7.7],
                 [8.8, 9.9, 10.1],
                 [11.2, 12.3, 13.4]]
    Output: B = [[5.5, 6.6, 7.7],
                 [8.8, 9.9, 10.1],
                 [11.2, 12.3, 13.4]]

### Constraints

- 1 ≤ `N` ≤ 4096
- All elements are 32-bit floating point numbers
- Performance is measured with `N` = 4,096

## CUDA

> To be developed.

## Triton

> To be developed.

## CuTe DSL

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/matrix-copy)
