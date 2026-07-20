# RGB to Grayscale

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Easy](../assets/common/difficulty_easy.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/rgb-to-grayscale)

Implement a GPU program that converts an RGB image to grayscale on the GPU. Given an input RGB image represented as a 1D array of 32-bit floating point values, compute the corresponding grayscale image using the standard RGB to grayscale conversion formula.

The conversion formula is: `gray = 0.299 × R + 0.587 × G + 0.114 × B`

The input array `input` contains `height × width × 3` elements, where the RGB values for each pixel are stored consecutively (R, G, B, R, G, B, ...). The output array `output` should contain `height × width` grayscale values.

### Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- Use the exact coefficients: 0.299 for red, 0.587 for green, 0.114 for blue

### Example 1:

    Input:  input = [255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 128.0, 128.0, 128.0], width=2, height=2
    Output: output = [76.245, 149.685, 29.07, 128.0]

### Example 2:

    Input:  input = [100.0, 150.0, 200.0], width=1, height=1
    Output: output = [140.75]

### Constraints

- 1 ≤ `width` ≤ 4096
- 1 ≤ `height` ≤ 4096
- `width × height` ≤ 4,194,304
- All RGB values are in the range \[0.0, 255.0\]
- Performance is measured with `height` = 2,048, `width` = 2,048

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/rgb-to-grayscale)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
