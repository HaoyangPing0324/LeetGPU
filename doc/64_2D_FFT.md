# 2D FFT

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/2d-fft)

Compute the 2D Discrete Fourier Transform (2D DFT) of a complex-valued signal stored on the GPU. Given a 2D complex input signal of shape `M × N`, compute its 2D DFT spectrum using the row-column decomposition: apply a 1D DFT along each row, then a 1D DFT along each column of the result. All values are 32-bit floating point.

### Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in `spectrum`
- The input and output are stored as 1D arrays of interleaved real and imaginary parts in row-major order: element `x[m, n]` has its real part at index `2*(m*N + n)` and imaginary part at index `2*(m*N + n) + 1`

### Example

Input: `M` = 2, `N` = 2  
Signal $x[m, n]$ (real part): $$ \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 0.0 \end{bmatrix} $$ Signal $x[m, n]$ (imaginary part): $$ \begin{bmatrix} 0.0 & 0.0 \\ 0.0 & 0.0 \end{bmatrix} $$ Output:  
Spectrum $X[k, l]$ (real part): $$ \begin{bmatrix} 1.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix} $$ Spectrum $X[k, l]$ (imaginary part): $$ \begin{bmatrix} 0.0 & 0.0 \\ 0.0 & 0.0 \end{bmatrix} $$

### Constraints

- 1 ≤ `M`, `N` ≤ 4096
- Signal values are 32-bit floating point (real and imaginary parts)
- Performance is measured with `M` = 2,048, `N` = 2,048

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/2d-fft)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
