# Nearest Neighbor

## Problem Description

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/nearest-neighbor)

Implement a GPU program that, for `N` three-dimensional points stored on the device, fills `indices[i]` with the index `j ≠ i` of the point closest to `points[i]`. Comparing *squared* Euclidean distance is sufficient—you do **not** need to compute square-roots.

### Implementation Requirements

- The `solve` function signature must remain unchanged
- External libraries are not permitted
- The final result must be stored in the `indices` array

### Example 1:

    Input:  points  = [(0,0,0), (1,0,0), (5,5,5)]
            indices = [-1, -1, -1]
            N       = 3
    Output: indices = [1, 0, 1]   # 0⇆1 are nearest, 2 is closest to 1

### Constraints

- 1 ≤ `N` ≤ 100,000
- Coordinates are 32-bit floats in the range \[-1000, 1000\]
- Performance is measured with `N` = 10,000

## CUDA

> To be developed.

## Triton

> To be developed.

## CuTe DSL

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/nearest-neighbor)
