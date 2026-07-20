# 3D Convolution

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/3d-convolution)

Implement a program that performs a 3D convolution operation. Given a 3D input volume and a 3D kernel (filter), compute the convolved output. The convolution should use a "valid" boundary condition (no padding).

For a 3D convolution, the output at position $(i,j,k)$ is given by:

$$ output(i,j,k) = \sum\_{d=0}^{K_d-1} \sum\_{r=0}^{K_r-1} \sum\_{c=0}^{K_c-1} input(i+d,j+r,k+c) \cdot kernel(d,r,c) $$

The input consists of:

- `input`: A 3D volume of 32-bit floats, as a 1D array (row-major, then depth).
- `kernel`: A 3D kernel of 32-bit floats, as a 1D array (row-major, then depth).
- `input_depth`, `input_rows`, `input_cols`: Dimensions of the input.
- `kernel_depth`, `kernel_rows`, `kernel_cols`: Dimensions of the kernel.

Output:

- `output`: A 1D array (row-major, then depth) storing the result.

Output dimensions:

- `output_depth = input_depth - kernel_depth + 1`
- `output_rows = input_rows - kernel_rows + 1`
- `output_cols = input_cols - kernel_cols + 1`

### Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in `output`

### Examples

#### Example 1:

Input volume $V \in \mathbb{R}^{3 \times 3 \times 3}$: $$ \begin{aligned} V\_{d=0} &= \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \\ V\_{d=1} &= \begin{bmatrix} 10 & 11 & 12 \\ 13 & 14 & 15 \\ 16 & 17 & 18 \end{bmatrix} \\ V\_{d=2} &= \begin{bmatrix} 19 & 20 & 21 \\ 22 & 23 & 24 \\ 25 & 26 & 27 \end{bmatrix} \end{aligned} $$ Kernel $K \in \mathbb{R}^{2 \times 3 \times 3}$: $$ \begin{aligned} K\_{d=0} &= \begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix} \\ K\_{d=1} &= \begin{bmatrix} 1 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \end{aligned} $$ Output $O \in \mathbb{R}^{2 \times 1 \times 1}$: $$ \[82, 163\] $$

#### Example 2:

Input volume $V \in \mathbb{R}^{2 \times 2 \times 2}$: $$ \begin{aligned} V\_{d=0} &= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \\ V\_{d=1} &= \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \end{aligned} $$ Kernel $K \in \mathbb{R}^{2 \times 2 \times 2}$: $$ \begin{aligned} K\_{d=0} &= \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \\ K\_{d=1} &= \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \end{aligned} $$ Output $O \in \mathbb{R}^{1 \times 1 \times 1}$: $$ \[36\] $$

### Constraints

- 1 ≤ `input_depth`, `input_rows`, `input_cols` ≤ 256
- 1 ≤ `kernel_depth`, `kernel_rows`, `kernel_cols` ≤ 5
- `kernel_depth` ≤ `input_depth`
- `kernel_rows` ≤ `input_rows`
- `kernel_cols` ≤ `input_cols`
- Performance is measured with `input_cols` = 128, `input_rows` = 128, `kernel_cols` = 5, `kernel_rows` = 5

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/3d-convolution)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
