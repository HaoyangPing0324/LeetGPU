# Vector Addition

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Easy](../assets/common/difficulty_easy.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/vector-addition)

Write a GPU program that performs element-wise addition of two vectors containing 32-bit floating point numbers. The program should take two input vectors of equal length and produce a single output vector containing their sum.

### Implementation Requirements

- External libraries are not permitted.
- The `solve` function signature must remain unchanged.
- The final result must be stored in vector `C`.

### Example 1

```text
Input:  A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]
```

### Example 2

```text
Input:  A = [1.5, 1.5, 1.5]
        B = [2.3, 2.3, 2.3]
Output: C = [3.8, 3.8, 3.8]
```

### Constraints

- Input vectors `A` and `B` have identical lengths.
- 1 <= `N` <= 100,000,000.
- Performance is measured with `N` = 25,000,000.

## CUDA

### Approach

CUDA assigns one GPU thread to each vector element.

- The global index is calculated with `blockIdx.x`, `blockDim.x`, and `threadIdx.x`.
- Each valid thread reads one value from `A` and `B`, adds them, and writes one value to `C`.
- Each block contains 256 threads, and ceiling division determines the number of blocks.
- The `idx < N` check prevents out-of-bounds access in the final block.
- The kernel is written in CUDA C++ and compiled with `nvcc`.

### Solution

```cuda
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

### Test Code

```cuda
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>

extern "C" void solve(const float* a, const float* b, float* c, int n);

int main() {
    constexpr int n = 4;
    const std::array<float, n> input_a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, n> input_b = {5.0f, 6.0f, 7.0f, 8.0f};
    const std::array<float, n> expected = {6.0f, 8.0f, 10.0f, 12.0f};
    std::array<float, n> actual = {};
    constexpr size_t bytes = n * sizeof(float);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    auto check_cuda = [](cudaError_t status, const char* operation) {
        if (status == cudaSuccess) {
            return true;
        }
        std::cerr << "Test failed. CUDA error in " << operation << ": "
                  << cudaGetErrorString(status) << '\n';
        return false;
    };

    bool success = check_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc(device_a)") &&
                   check_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc(device_b)") &&
                   check_cuda(cudaMalloc(&device_c, bytes), "cudaMalloc(device_c)") &&
                   check_cuda(cudaMemcpy(device_a, input_a.data(), bytes, cudaMemcpyHostToDevice), "copy input_a") &&
                   check_cuda(cudaMemcpy(device_b, input_b.data(), bytes, cudaMemcpyHostToDevice), "copy input_b");

    if (success) {
        solve(device_a, device_b, device_c, n);
        success = check_cuda(cudaGetLastError(), "solve") &&
                  check_cuda(cudaDeviceSynchronize(), "synchronize") &&
                  check_cuda(cudaMemcpy(actual.data(), device_c, bytes, cudaMemcpyDeviceToHost), "copy result");
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    if (!success) {
        return 1;
    }

    for (int index = 0; index < n; ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1e-6f) {
            std::cerr << "Test failed at index " << index
                      << ". Expected: " << expected[index]
                      << ", Actual: " << actual[index] << '\n';
            return 1;
        }
    }

    std::cout << "Test passed.\n";
    return 0;
}
```

### Test Result

| Platform | Status | Problem Size | Iterations | Average Time | Minimum Time | Maximum Time | Performance |
|---|:---:|---|---:|---:|---:|---:|---:|
| NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | N=25000000 | 20 | 0.929 ms | 0.727 ms | 1.708 ms | 322.986 GB/s |
| NVIDIA GeForce RTX 4090 | PASS | N=25000000 | 20 | 0.379 ms | 0.327 ms | 1.332 ms | 792.152 GB/s |

## Triton

### Approach

Triton assigns a block of vector elements to each program instance.

- `tl.program_id(0)` identifies the current program instance.
- Each instance processes up to 1024 elements generated with `tl.arange`.
- The kernel loads two blocks, adds them element by element, and stores the result.
- A mask prevents out-of-bounds loads and stores in the final block.
- The kernel is written in Python and JIT-compiled by Triton for the GPU.

### Solution

```python
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask=offsets<n_elements

    a_block = tl.load(a+offsets, mask=mask)
    b_block = tl.load(b+offsets, mask=mask)

    c_block = a_block+b_block

    tl.store(c+offsets, c_block, mask=mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
```

### Test Code

```python
import importlib.util
import sys
from pathlib import Path

import torch


def load_implementation():
    source_path = Path(__file__).resolve().parents[2] / "src" / "triton" / "01_Vector_Addition.py"
    spec = importlib.util.spec_from_file_location("vector_addition_triton_minimal", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    input_a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    input_b = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda")
    expected = torch.tensor([6.0, 8.0, 10.0, 12.0], device="cuda")
    actual = torch.empty_like(input_a)

    load_implementation().solve(input_a, input_b, actual, input_a.numel())
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)
    print("Test passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Test failed: {error}", file=sys.stderr)
        raise
```

### Test Result

| Platform | Status | Problem Size | Iterations | Average Time | Minimum Time | Maximum Time | Performance |
|---|:---:|---|---:|---:|---:|---:|---:|
| NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | N=25000000 | 20 | 0.965 ms | 0.746 ms | 1.754 ms | 310.909 GB/s |
| NVIDIA GeForce RTX 4090 | PASS | N=25000000 | 20 | 0.344 ms | 0.339 ms | 0.356 ms | 872.288 GB/s |


## PyTorch

### Approach

PyTorch performs the vector addition with the built-in `torch.add` operation.

- `torch.add(A, B, out=C)` adds corresponding elements and writes directly to `C`.
- PyTorch selects and launches the appropriate GPU kernel automatically.
- Thread scheduling, indexing, and bounds handling are managed internally by PyTorch.
- `N` is kept in the required interface but is not used directly by this implementation.

### Solution

```python
import torch


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    torch.add(A,B,out = C);
```

### Test Code

```python
import importlib.util
import sys
from pathlib import Path

import torch


def load_implementation():
    source_path = Path(__file__).resolve().parents[2] / "src" / "pytorch" / "01_Vector_Addition.py"
    spec = importlib.util.spec_from_file_location("vector_addition_pytorch_minimal", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    input_a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    input_b = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda")
    expected = torch.tensor([6.0, 8.0, 10.0, 12.0], device="cuda")
    actual = torch.empty_like(input_a)

    load_implementation().solve(input_a, input_b, actual, input_a.numel())
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)
    print("Test passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Test failed: {error}", file=sys.stderr)
        raise
```

### Test Result

| Platform | Status | Problem Size | Iterations | Average Time | Minimum Time | Maximum Time | Performance |
|---|:---:|---|---:|---:|---:|---:|---:|
| NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | N=25000000 | 20 | 0.788 ms | 0.744 ms | 1.087 ms | 380.604 GB/s |
| NVIDIA GeForce RTX 4090 | PASS | N=25000000 | 20 | 0.331 ms | 0.330 ms | 0.339 ms | 905.015 GB/s |

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/vector-addition)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
