# K-Means Clustering

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Hard](../assets/common/difficulty_hard.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/k-means-clustering)

Implement the k-means clustering algorithm for 2D points. Given arrays of x and y coordinates for data points, initial centroids, and other parameters, assign each point to the nearest centroid and update the centroids iteratively. The final centroids and labels should be stored in the output arrays.

### Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in `labels`, `final_centroid_x`, and `final_centroid_y`

### Example 1:

    Input:
    sample_size = 4, k = 2, max_iterations = 10
    data_x = [1.0, 2.0, 8.0, 9.0]
    data_y = [1.0, 2.0, 8.0, 9.0]
    initial_centroid_x = [1.0, 8.0]
    initial_centroid_y = [1.0, 8.0]
    Output: (see reference implementation for expected output)

### Constraints

- 1 ≤ sample_size ≤ 1000000
- 1 ≤ k ≤ 1000
- All arrays are float32 except labels, which is int32
- Performance is measured with `k` = 5, `max_iterations` = 30, `sample_size` = 10,000

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/k-means-clustering)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
