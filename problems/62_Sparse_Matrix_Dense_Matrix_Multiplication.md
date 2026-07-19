# Sparse Matrix-Dense Matrix Multiplication

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[Original LeetGPU Challenge](https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication)

Implement a GPU program that multiplies a sparse matrix `A` of dimensions `M` × `N` by a dense matrix `B` of dimensions `N` × `K`, producing a dense output matrix `C` of dimensions `M` × `K`. All matrices are stored in row-major order using 32-bit floats. The matrix `A` is approximately 60–70% sparse (i.e., 60–70% of elements are zero), and `nnz` gives the number of non-zero elements in `A`.

Mathematically, the operation is defined as: $$ C\_{ij} = \sum\_{k=0}^{N-1} A\_{ik} \cdot B\_{kj} \quad \text{for} \quad i = 0, \ldots, M-1,\\ j = 0, \ldots, K-1 $$

### Implementation Requirements

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in matrix `C`

### Example

Input:  
Matrix $A$ ($3 \times 4$): $$ \begin{bmatrix} 2.0 & 0.0 & 0.0 & 1.0 \\ 0.0 & 3.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 4.0 & 0.0 \end{bmatrix} $$ Matrix $B$ ($4 \times 2$): $$ \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \\ 5.0 & 6.0 \\ 7.0 & 8.0 \end{bmatrix} $$ Output:  
Matrix $C$ ($3 \times 2$): $$ \begin{bmatrix} 9.0 & 12.0 \\ 9.0 & 12.0 \\ 20.0 & 24.0 \end{bmatrix} $$

### Constraints

- 1 ≤ `M`, `N`, `K` ≤ 8,192
- All values in `A` and `B` are 32-bit floats in the range \[−10, 10\]
- The matrix `A` is approximately 60–70% sparse
- Performance is measured with `M` = 4,096, `N` = 2,048, `K` = 512
