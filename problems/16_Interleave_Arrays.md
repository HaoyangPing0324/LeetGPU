# Interleave Arrays

![Difficulty: Easy](../assets/common/difficulty_easy.svg)

[Original LeetGPU Challenge](https://leetgpu.com/challenges/interleave-arrays)

Write a GPU program that interleaves two arrays of 32-bit floating point numbers. Given two input arrays `A` and `B`, each of length `N`, produce an output array of length `2N` where elements alternate between the two inputs: `[A[0], B[0], A[1], B[1], A[2], B[2], ...]`

![](../assets/16_Interleave_Arrays/image_01.svg)

### Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` array

### Example 1:

    Input:  A = [1.0, 2.0, 3.0], B = [4.0, 5.0, 6.0]
    Output: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

### Example 2:

    Input:  A = [10.0, 20.0], B = [30.0, 40.0]
    Output: [10.0, 30.0, 20.0, 40.0]

### Constraints

- 1 ≤ `N` ≤ 50,000,000
- Performance is measured with `N` = 25,000,000
