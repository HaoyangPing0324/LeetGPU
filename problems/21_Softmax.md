# Softmax

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[Original LeetGPU Challenge](https://leetgpu.com/challenges/softmax)

Write a program that computes the softmax function for an array of 32-bit floating-point numbers on a GPU. The softmax function is defined as follows:

For an input array $x$ of length $n$, the softmax of $x$, denoted $\sigma(x)$, is an array of length $n$ where the $i$-th element is:

$\sigma(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$

Your solution should handle potential overflow issues by using the "max trick". Subtract the maximum value of the input array from each element before exponentiation.

### Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`

### Example 1:

    Input: [1.0, 2.0, 3.0], N = 3
    Output: [0.090, 0.244, 0.665] (approximately)

### Example 2:

    Input: [-10.0, -5.0, 0.0, 5.0, 10.0], N = 5
    Output: [2.047e-09, 3.038e-07, 4.509e-05, 6.693e-03, 9.933e-01] (approximately)

### Constraints

- 1 ≤ `N` ≤ 500,000
- Performance is measured with `N` = 500,000
