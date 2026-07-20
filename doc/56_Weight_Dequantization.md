# Weight Dequantization

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/weight-dequantization)

Implement a GPU program that "dequantizes" a weight matrix on the GPU. You are given an input matrix `X` of shape `[M, N]` containing quantized values and a scale matrix `S` of shape `[ceil(M/T), ceil(N/T)]`, where `T` is the tile size.

For each element $X_{i,j}$, the corresponding scale factor is $S_{row, col}$ where $row = \lfloor i / T \rfloor$ and $col = \lfloor j / T \rfloor$. The output $Y_{i,j}$ should be computed as: $$ Y\_{i,j} = X\_{i,j} \times S\_{row, col} $$

### Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the output buffer `Y`

### Example 1:

    Input:
    M = 4, N = 4, TILE_SIZE = 2
    X = [
      [10, 10,  5,  5],
      [10, 10,  5,  5],
      [ 2,  2,  8,  8],
      [ 2,  2,  8,  8]
    ]
    S = [
      [0.5, 2.0],
      [4.0, 0.25]
    ]

    Output:
    Y = [
      [ 5.0,  5.0, 10.0, 10.0],
      [ 5.0,  5.0, 10.0, 10.0],
      [ 8.0,  8.0,  2.0,  2.0],
      [ 8.0,  8.0,  2.0,  2.0]
    ]
    Explanation:
    Tile (0,0) of X is multiplied by S[0,0] (0.5).
    Tile (0,1) of X is multiplied by S[0,1] (2.0).
    Tile (1,0) is multiplied by S[1,0] (4.0).
    Tile (1,1) is multiplied by S[1,1] (0.25).

### Constraints

- 1 ≤ `M`, `N` ≤ 8192
- `TILE_SIZE` ∈ {16, 32, 64, 128}
- Performance is measured with `M` = 8,192, `N` = 8,192, `TILE_SIZE` = 128

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/weight-dequantization)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
