# Rotary Positional Embedding

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/rotary-positional-embedding)

Implement a GPU program that computes the Rotary Positional Embedding (RoPE) for a batch of query vectors. RoPE is a method for encoding positional information in transformer models by rotating the query and key vectors using precomputed cosine and sine components.

Mathematically, given a query vector $x$ and corresponding cosine and sine vectors, the operation is defined as: $$ \text{RoPE}(x) = x \odot \cos + \text{rotate\\half}(x) \odot \sin $$

Where $\odot$ denotes element-wise multiplication. The $\text{rotate\_half}(x)$ operation swaps the first and second halves of the vector and negates the first half. For a vector of dimension $d$: $$ \text{rotate\\half}(\[x_1, \dots, x\_{d/2}, x\_{d/2+1}, \dots, x_d\]) = \[-x\_{d/2+1}, \dots, -x_d, x_1, \dots, x\_{d/2}\] $$

### Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The input tensors `Q`, `cos`, and `sin` have shape `(M, D)`, where `M` is the number of tokens and `D` is the head dimension
- `D` (head dimension) is guaranteed to be an even number
- The final result must be stored in the output variable with the same shape `(M, D)`

### Example 1:

    Input:  Q   = [[1.0, 2.0, 3.0, 4.0],
                   [1.0, 1.0, 1.0, 1.0]]
            Cos = [[1.0, 1.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 0.0]]
            Sin = [[0.0, 0.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0, 1.0]]
    Output: result = [[1.0, 2.0, 3.0, 4.0],
                      [-1.0, -1.0, 1.0, 1.0]]
            (Row 0 is identity via Cos; Row 1 is rotated via Sin)

### Constraints

- `Q`, `cos`, and `sin` have identical dimensions
- `D` % 2 == 0
- 1 ≤ `M`, `D` ≤ 10,000
- Performance is measured with `D` = 128, `M` = 1,048,576

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/rotary-positional-embedding)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
