# Matrix Multiplication

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Easy](../assets/common/difficulty_easy.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/matrix-multiplication)

Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix \(A\) of dimensions \(M \times N\) and matrix \(B\) of dimensions \(N \times K\), compute the product matrix \(C = A \times B\), which will have dimensions \(M \times K\). All matrices are stored in row-major format.

### Implementation Requirements

- Use only native features (external libraries are not permitted).
- The `solve` function signature must remain unchanged.
- The final result must be stored in matrix `C`.

### Example 1

Input:

Matrix \(A\) (\(2 \times 2\)):

\[
\begin{bmatrix}
1.0 & 2.0 \\
3.0 & 4.0
\end{bmatrix}
\]

Matrix \(B\) (\(2 \times 2\)):

\[
\begin{bmatrix}
5.0 & 6.0 \\
7.0 & 8.0
\end{bmatrix}
\]

Output:

Matrix \(C\) (\(2 \times 2\)):

\[
\begin{bmatrix}
19.0 & 22.0 \\
43.0 & 50.0
\end{bmatrix}
\]

### Example 2

Input:

Matrix \(A\) (\(1 \times 3\)):

\[
\begin{bmatrix}
1.0 & 2.0 & 3.0
\end{bmatrix}
\]

Matrix \(B\) (\(3 \times 1\)):

\[
\begin{bmatrix}
4.0 \\
5.0 \\
6.0
\end{bmatrix}
\]

Output:

Matrix \(C\) (\(1 \times 1\)):

\[
\begin{bmatrix}
32.0
\end{bmatrix}
\]

### Constraints

- 1 <= `M`, `N`, `K` <= 8192.
- Performance is measured with `M` = 8192, `N` = 6144, `K` = 4096.

All reported runs were launched from a Linux client. The NVIDIA GeForce RTX 5070 Ti Laptop GPU results were collected on a Windows remote server, while the NVIDIA GeForce RTX 4090 results were collected on a Linux remote server.

## CUDA

### Approach

The CUDA solution is organized as an optimization sequence. Every version implements the same LeetGPU contract, `A[M, N] x B[N, K] = C[M, K]`, and is checked with the same correctness cases. The versions differ only in how work and data movement are organized.

#### V0: Naive

Each two-dimensional CUDA thread computes one output element. It reads one complete row of `A` and one complete column of `B` directly from global memory. This is the clearest mapping from the matrix definition, but adjacent threads repeatedly load values that could have been shared.

#### V1: Original Shared-Memory Tiling

This is the project's original implementation. A `16 x 16` block computes a `16 x 16` output tile. Threads cooperatively stage one tile from each input in shared memory, synchronize, and reuse those values for 16 multiply-add operations. Zero-filled boundary loads support dimensions that are not multiples of 16.

#### V1: One-Dimensional Thread Tiling

A block computes a `64 x 64` output tile. Instead of producing one result, each thread accumulates 16 output rows for one output column. This increases register reuse of each loaded `B` value and reduces the number of threads required for the same output area.

#### V1: Two-Dimensional Thread Tiling

Each thread accumulates an `8 x 4` micro-tile of `C`. Values from shared memory are first loaded into small register arrays, then reused across the 32 outputs owned by the thread. This raises arithmetic work per shared-memory access and substantially improves throughput.

#### V2: Vectorized Memory Access

The two-dimensional register tile is retained, while aligned global and shared-memory transfers use `float4`. Four adjacent values are moved by one vector operation. The optimized path is used only when the reduction and output-column dimensions are multiples of four; other shapes use a safe scalar fallback, so the LeetGPU boundary cases remain valid.

#### V3: Double Buffering

Two shared-memory buffers are used alternately. While the current tile is consumed, the next tile is prefetched into registers. After computation, the prefetched values are committed to the other shared-memory buffer. This overlaps part of global-memory latency with arithmetic, although the measured gain depends on register pressure, synchronization, and tile shape.

#### V4: Larger Output Tile

This project-specific experiment expands the output block from `64 x 64` to `128 x 128` while retaining an `8 x 4` thread micro-tile. It improves the verified 5070 Ti result but is slower on the 4090, demonstrating that a larger tile is not universally better.

Three additional complete runs of V3 and V4 were performed on each GPU. V3 remained consistently faster on the RTX 4090, while V4 remained consistently faster on the RTX 5070 Ti Laptop GPU. The project therefore keeps V3 as the simple default and presents V4 as a platform-sensitive experiment rather than hard-coding a device-specific dispatch rule.

The active implementation is selected in `02_Matrix_Multiplication.cu` by leaving exactly one include uncommented. The reusable run scripts may also select a method explicitly without editing the selector:

```text
bash scripts/run_cuda_remote.sh 02_Matrix_Multiplication <method>
```

### Solution

The selector enables the default version. Expand any method below to view its complete source code.

<details>
<summary>Default Selector</summary>

```cuda
// Select exactly one implementation.
// #include "02_Matrix_Multiplication_v0_naive.cu"
// #include "02_Matrix_Multiplication_v1_shared_memory.cu"
// #include "02_Matrix_Multiplication_v1_1d_thread_tiling.cu"
// #include "02_Matrix_Multiplication_v1_2d_thread_tiling.cu"
// #include "02_Matrix_Multiplication_v2_vectorized.cu"
#include "02_Matrix_Multiplication_v3_double_buffered.cu"
// #include "02_Matrix_Multiplication_v4_large_tile.cu"
```

</details>

<details>
<summary>V0 Naive</summary>

```cuda
#include <cuda_runtime.h>

__global__ void matrix_multiplication_v0(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (int inner = 0; inner < N; ++inner) {
        sum += A[row * N + inner] * B[inner * K + col];
    }
    C[row * K + col] = sum;
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    const dim3 block(16, 16);
    const dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matrix_multiplication_v0<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

</details>

<details>
<summary>V1 Original Shared Memory</summary>

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE  16
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float S = 0.0f;
    for (int i = 0 ; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE ; i++){
        int col2 = tx + i * BLOCK_SIZE;
        int row2 = ty + i * BLOCK_SIZE;
        A_shared[ty][tx] = (row < M && col2 < N) ? A[row * N + col2] : 0.0f;
        B_shared[ty][tx] = (row2 < N && col < K) ? B[row2 * K + col] : 0.0f;
        __syncthreads();
        for (int j = 0 ; j <  BLOCK_SIZE ; j++){
            S += A_shared[ty][j] * B_shared[j][tx];
        }
        __syncthreads();
    }
    if(row < M && col < K){
        C[row * K + col] = S ;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

</details>

<details>
<summary>V1 1D Thread Tiling</summary>

```cuda
#include <cuda_runtime.h>

template <int BM = 64, int BN = 64, int BK = 16, int TM = 16>
__global__ void matrix_multiplication_v1_1d(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K) {
    constexpr int NT = (BM / TM) * BN;
    __shared__ float SA[BM][BK];
    __shared__ float SB[BK][BN];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int col = blockIdx.x * BN + tx;
    float sum[TM] = {0.0f};

    for (int tile = 0; tile < N; tile += BK) {
        #pragma unroll
        for (int i = tid; i < BM * BK; i += NT) {
            const int local_row = i / BK;
            const int local_col = i % BK;
            const int row = blockIdx.y * BM + local_row;
            const int inner = tile + local_col;
            SA[local_row][local_col] =
                (row < M && inner < N) ? A[row * N + inner] : 0.0f;
        }
        #pragma unroll
        for (int i = tid; i < BK * BN; i += NT) {
            const int local_row = i / BN;
            const int local_col = i % BN;
            const int inner = tile + local_row;
            const int output_col = blockIdx.x * BN + local_col;
            SB[local_row][local_col] =
                (inner < N && output_col < K) ? B[inner * K + output_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
            const float b = SB[inner][tx];
            #pragma unroll
            for (int row = 0; row < TM; ++row) {
                sum[row] += SA[ty * TM + row][inner] * b;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int local_row = 0; local_row < TM; ++local_row) {
        const int row = blockIdx.y * BM + ty * TM + local_row;
        if (row < M && col < K) C[row * K + col] = sum[local_row];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 16, TM = 16;
    const dim3 block(BN, BM / TM);
    const dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    matrix_multiplication_v1_1d<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

</details>

<details>
<summary>V1 2D Thread Tiling</summary>

```cuda
#include <cuda_runtime.h>

template <int BM = 64, int BN = 64, int BK = 16, int TM = 8, int TN = 4>
__global__ void matrix_multiplication_v1_2d(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K) {
    constexpr int NT = (BM / TM) * (BN / TN);
    __shared__ float SA[BM][BK];
    __shared__ float SB[BK][BN];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    float sum[TM][TN] = {{0.0f}};

    for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
        const int base_inner = tile * BK;
        #pragma unroll
        for (int i = tid; i < BM * BK; i += NT) {
            const int local_row = i / BK;
            const int local_col = i % BK;
            const int row = blockIdx.y * BM + local_row;
            const int inner = base_inner + local_col;
            SA[local_row][local_col] =
                (row < M && inner < N) ? A[row * N + inner] : 0.0f;
        }
        #pragma unroll
        for (int i = tid; i < BK * BN; i += NT) {
            const int local_row = i / BN;
            const int local_col = i % BN;
            const int inner = base_inner + local_row;
            const int col = blockIdx.x * BN + local_col;
            SB[local_row][local_col] =
                (inner < N && col < K) ? B[inner * K + col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int row = 0; row < TM; ++row) a_reg[row] = SA[ty * TM + row][inner];
            #pragma unroll
            for (int col = 0; col < TN; ++col) b_reg[col] = SB[inner][tx * TN + col];
            #pragma unroll
            for (int row = 0; row < TM; ++row)
                #pragma unroll
                for (int col = 0; col < TN; ++col)
                    sum[row][col] += a_reg[row] * b_reg[col];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int local_row = 0; local_row < TM; ++local_row) {
        const int row = blockIdx.y * BM + ty * TM + local_row;
        if (row >= M) continue;
        #pragma unroll
        for (int local_col = 0; local_col < TN; ++local_col) {
            const int col = blockIdx.x * BN + tx * TN + local_col;
            if (col < K) C[row * K + col] = sum[local_row][local_col];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 16, TM = 8, TN = 4;
    const dim3 block(BN / TN, BM / TM);
    const dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    matrix_multiplication_v1_2d<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

</details>

<details>
<summary>V2 Vectorized</summary>

```cuda
#include <cuda_runtime.h>

__global__ void scalar_fallback(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;
    float sum = 0.0f;
    for (int inner = 0; inner < N; ++inner) {
        sum += A[row * N + inner] * B[inner * K + col];
    }
    C[row * K + col] = sum;
}

template <int BM = 64, int BN = 64, int BK = 32, int TM = 8, int TN = 4>
__global__ void matrix_multiplication_v2(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    static_assert(BK % 4 == 0 && BN % 4 == 0 && TN % 4 == 0,
                  "Vectorized dimensions must be multiples of four");
    constexpr int NT = (BM / TM) * (BN / TN);
    __shared__ float SA[BM][BK];
    __shared__ float SB[BK][BN];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    float sum[TM][TN] = {{0.0f}};

    for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
        const int base_inner = tile * BK;
        #pragma unroll
        for (int i = tid; i < BM * BK / 4; i += NT) {
            const int local_row = i / (BK / 4);
            const int local_col = (i % (BK / 4)) * 4;
            const int row = blockIdx.y * BM + local_row;
            const int inner = base_inner + local_col;
            float4 value = make_float4(0, 0, 0, 0);
            if (row < M && inner + 3 < N) {
                value = *reinterpret_cast<const float4*>(&A[row * N + inner]);
            }
            *reinterpret_cast<float4*>(&SA[local_row][local_col]) = value;
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 4; i += NT) {
            const int local_row = i / (BN / 4);
            const int local_col = (i % (BN / 4)) * 4;
            const int inner = base_inner + local_row;
            const int col = blockIdx.x * BN + local_col;
            float4 value = make_float4(0, 0, 0, 0);
            if (inner < N && col + 3 < K) {
                value = *reinterpret_cast<const float4*>(&B[inner * K + col]);
            }
            *reinterpret_cast<float4*>(&SB[local_row][local_col]) = value;
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
            const float4 b = *reinterpret_cast<const float4*>(&SB[inner][tx * TN]);
            #pragma unroll
            for (int row = 0; row < TM; ++row) {
                const float a = SA[ty * TM + row][inner];
                sum[row][0] += a * b.x;
                sum[row][1] += a * b.y;
                sum[row][2] += a * b.z;
                sum[row][3] += a * b.w;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int local_row = 0; local_row < TM; ++local_row) {
        const int row = blockIdx.y * BM + ty * TM + local_row;
        const int col = blockIdx.x * BN + tx * TN;
        if (row < M && col + 3 < K) {
            *reinterpret_cast<float4*>(&C[row * K + col]) =
                make_float4(sum[local_row][0], sum[local_row][1],
                            sum[local_row][2], sum[local_row][3]);
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    if ((N & 3) != 0 || (K & 3) != 0) {
        const dim3 block(16, 16);
        const dim3 grid((K + 15) / 16, (M + 15) / 16);
        scalar_fallback<<<grid, block>>>(A, B, C, M, N, K);
    } else {
        constexpr int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
        const dim3 block(BN / TN, BM / TM);
        const dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
        matrix_multiplication_v2<BM, BN, BK, TM, TN>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}
```

</details>

<details>
<summary>V3 Double Buffered</summary>

```cuda
#include <cuda_runtime.h>

#ifndef MATMUL_BM
#define MATMUL_BM 64
#define MATMUL_BN 64
#define MATMUL_BK 16
#define MATMUL_TM 8
#define MATMUL_TN 4
#endif

__global__ void scalar_fallback_v3(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;
    float sum = 0.0f;
    for (int inner = 0; inner < N; ++inner) {
        sum += A[row * N + inner] * B[inner * K + col];
    }
    C[row * K + col] = sum;
}

template <int BM = 64, int BN = 64, int BK = 16, int TM = 8, int TN = 4>
__global__ void matrix_multiplication_v3(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    constexpr int NT = (BM / TM) * (BN / TN);
    constexpr int A_FLOAT4_PER_THREAD = (BM * BK / 4 + NT - 1) / NT;
    constexpr int B_FLOAT4_PER_THREAD = (BK * BN / 4 + NT - 1) / NT;
    __shared__ float SA[2][BM][BK];
    __shared__ float SB[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    float a_prefetch[A_FLOAT4_PER_THREAD][4];
    float b_prefetch[B_FLOAT4_PER_THREAD][4];
    float sum[TM][TN] = {{0.0f}};

    auto load_tile = [&](int tile) {
        const int base_inner = tile * BK;
        #pragma unroll
        for (int i = 0; i < A_FLOAT4_PER_THREAD; ++i) {
            const int index = (i * NT + tid) * 4;
            float4 value = make_float4(0, 0, 0, 0);
            if (index < BM * BK) {
                const int local_row = index / BK;
                const int local_col = index % BK;
                const int row = blockIdx.y * BM + local_row;
                const int inner = base_inner + local_col;
                if (row < M && inner + 3 < N) {
                    value = *reinterpret_cast<const float4*>(&A[row * N + inner]);
                }
            }
            a_prefetch[i][0] = value.x; a_prefetch[i][1] = value.y;
            a_prefetch[i][2] = value.z; a_prefetch[i][3] = value.w;
        }
        #pragma unroll
        for (int i = 0; i < B_FLOAT4_PER_THREAD; ++i) {
            const int index = (i * NT + tid) * 4;
            float4 value = make_float4(0, 0, 0, 0);
            if (index < BK * BN) {
                const int local_row = index / BN;
                const int local_col = index % BN;
                const int inner = base_inner + local_row;
                const int col = blockIdx.x * BN + local_col;
                if (inner < N && col + 3 < K) {
                    value = *reinterpret_cast<const float4*>(&B[inner * K + col]);
                }
            }
            b_prefetch[i][0] = value.x; b_prefetch[i][1] = value.y;
            b_prefetch[i][2] = value.z; b_prefetch[i][3] = value.w;
        }
    };

    auto store_tile = [&](int buffer) {
        #pragma unroll
        for (int i = 0; i < A_FLOAT4_PER_THREAD; ++i) {
            const int index = (i * NT + tid) * 4;
            if (index < BM * BK) {
                const int local_row = index / BK;
                const int local_col = index % BK;
                *reinterpret_cast<float4*>(&SA[buffer][local_row][local_col]) =
                    make_float4(a_prefetch[i][0], a_prefetch[i][1],
                                a_prefetch[i][2], a_prefetch[i][3]);
            }
        }
        #pragma unroll
        for (int i = 0; i < B_FLOAT4_PER_THREAD; ++i) {
            const int index = (i * NT + tid) * 4;
            if (index < BK * BN) {
                const int local_row = index / BN;
                const int local_col = index % BN;
                *reinterpret_cast<float4*>(&SB[buffer][local_row][local_col]) =
                    make_float4(b_prefetch[i][0], b_prefetch[i][1],
                                b_prefetch[i][2], b_prefetch[i][3]);
            }
        }
    };

    const int tiles = (N + BK - 1) / BK;
    load_tile(0);
    store_tile(0);
    __syncthreads();

    int read_buffer = 0;
    for (int tile = 0; tile < tiles; ++tile) {
        if (tile + 1 < tiles) load_tile(tile + 1);

        #pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
            #pragma unroll
            for (int local_col = 0; local_col < TN; local_col += 4) {
                const float4 b = *reinterpret_cast<const float4*>(
                    &SB[read_buffer][inner][tx * TN + local_col]);
                #pragma unroll
                for (int row = 0; row < TM; ++row) {
                    const float a = SA[read_buffer][ty * TM + row][inner];
                    sum[row][local_col + 0] += a * b.x;
                    sum[row][local_col + 1] += a * b.y;
                    sum[row][local_col + 2] += a * b.z;
                    sum[row][local_col + 3] += a * b.w;
                }
            }
        }
        __syncthreads();

        if (tile + 1 < tiles) {
            const int write_buffer = read_buffer ^ 1;
            store_tile(write_buffer);
            __syncthreads();
            read_buffer = write_buffer;
        }
    }

    #pragma unroll
    for (int local_row = 0; local_row < TM; ++local_row) {
        const int row = blockIdx.y * BM + ty * TM + local_row;
        #pragma unroll
        for (int local_col = 0; local_col < TN; local_col += 4) {
            const int col = blockIdx.x * BN + tx * TN + local_col;
            if (row < M && col + 3 < K) {
                *reinterpret_cast<float4*>(&C[row * K + col]) =
                    make_float4(sum[local_row][local_col + 0],
                                sum[local_row][local_col + 1],
                                sum[local_row][local_col + 2],
                                sum[local_row][local_col + 3]);
            }
        }
    }
}

#ifndef MATMUL_NO_SOLVE
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    if ((N & 3) != 0 || (K & 3) != 0) {
        const dim3 block(16, 16);
        const dim3 grid((K + 15) / 16, (M + 15) / 16);
        scalar_fallback_v3<<<grid, block>>>(A, B, C, M, N, K);
    } else {
        constexpr int BM = MATMUL_BM, BN = MATMUL_BN, BK = MATMUL_BK;
        constexpr int TM = MATMUL_TM, TN = MATMUL_TN;
        const dim3 block(BN / TN, BM / TM);
        const dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
        matrix_multiplication_v3<BM, BN, BK, TM, TN>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}
#endif
```

</details>

<details>
<summary>V4 Large Tile</summary>

```cuda
#define MATMUL_BM 128
#define MATMUL_BN 128
#define MATMUL_BK 16
#define MATMUL_TM 8
#define MATMUL_TN 4

#include "02_Matrix_Multiplication_v3_double_buffered.cu"
```

</details>

### Test Code

```cuda
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>

extern "C" void solve(const float* a, const float* b, float* c,
                      int m, int n, int k);

int main() {
    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int k = 2;
    const std::array<float, 4> matrix_a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> matrix_b = {5.0f, 6.0f, 7.0f, 8.0f};
    const std::array<float, 4> expected = {19.0f, 22.0f, 43.0f, 50.0f};
    std::array<float, 4> actual = {};
    constexpr size_t bytes = 4 * sizeof(float);

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
                   check_cuda(cudaMemcpy(device_a, matrix_a.data(), bytes,
                                         cudaMemcpyHostToDevice), "copy matrix_a") &&
                   check_cuda(cudaMemcpy(device_b, matrix_b.data(), bytes,
                                         cudaMemcpyHostToDevice), "copy matrix_b");

    if (success) {
        solve(device_a, device_b, device_c, m, n, k);
        success = check_cuda(cudaGetLastError(), "solve") &&
                  check_cuda(cudaDeviceSynchronize(), "synchronize") &&
                  check_cuda(cudaMemcpy(actual.data(), device_c, bytes,
                                        cudaMemcpyDeviceToHost), "copy result");
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    if (!success) {
        return 1;
    }

    for (size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1e-5f) {
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

All methods use `M=8192, N=6144, K=4096` and three measured iterations.

| Method | Platform | Status | Average Time | Minimum Time | Maximum Time | Performance |
|---|---|:---:|---:|---:|---:|---:|
| V0 Naive | NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | 348.822 ms | 303.296 ms | 408.415 ms | 1182.026 GFLOPS |
| V0 Naive | NVIDIA GeForce RTX 4090 | PASS | 83.949 ms | 83.063 ms | 85.607 ms | 4911.543 GFLOPS |
| V1 Original Shared Memory | NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | 242.778 ms | 215.807 ms | 296.659 ms | 1698.330 GFLOPS |
| V1 Original Shared Memory | NVIDIA GeForce RTX 4090 | PASS | 61.351 ms | 59.362 ms | 65.267 ms | 6720.632 GFLOPS |
| V1 1D Thread Tiling | NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | 60.158 ms | 59.850 ms | 60.709 ms | 6853.909 GFLOPS |
| V1 1D Thread Tiling | NVIDIA GeForce RTX 4090 | PASS | 19.508 ms | 19.498 ms | 19.514 ms | 21135.588 GFLOPS |
| V1 2D Thread Tiling | NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | 45.858 ms | 44.093 ms | 48.239 ms | 8991.259 GFLOPS |
| V1 2D Thread Tiling | NVIDIA GeForce RTX 4090 | PASS | 12.540 ms | 12.421 ms | 12.771 ms | 32880.130 GFLOPS |
| V2 Vectorized | NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | 33.253 ms | 31.828 ms | 34.445 ms | 12399.341 GFLOPS |
| V2 Vectorized | NVIDIA GeForce RTX 4090 | PASS | 8.642 ms | 8.641 ms | 8.643 ms | 47711.609 GFLOPS |
| V3 Double Buffered | NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | 33.903 ms | 30.154 ms | 37.208 ms | 12161.818 GFLOPS |
| V3 Double Buffered | NVIDIA GeForce RTX 4090 | PASS | 8.740 ms | 8.738 ms | 8.742 ms | 47178.031 GFLOPS |
| V4 Large Tile | NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | 29.029 ms | 28.421 ms | 29.768 ms | 14203.795 GFLOPS |
| V4 Large Tile | NVIDIA GeForce RTX 4090 | PASS | 9.454 ms | 9.017 ms | 10.093 ms | 43611.750 GFLOPS |

## Triton

### Approach

#### The same matrix product at a different abstraction level

The Triton implementation computes the same relationship as the CUDA kernel:

\[
C[m,k] = \sum_{n=0}^{N-1} A[m,n]B[n,k].
\]

The main difference is the unit of parallel work. CUDA explicitly assigns one output element to one thread and manually stages data in shared memory. Triton assigns an entire output tile to one program instance and describes its work with vectors and small tensor blocks. Triton's compiler is then responsible for lowering the block operations to GPU threads, registers, memory operations, and synchronization.

#### Two-dimensional program grid

All three compile-time block sizes are fixed to 32:

```text
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32
```

Here, `M` and `K` identify the two output dimensions, while `N` is the reduction dimension. The launch grid is:

```text
(ceil(M / 32), ceil(K / 32))
```

`tl.program_id(0)` selects a block of 32 output rows and `tl.program_id(1)` selects a block of 32 output columns. Consequently, one program instance owns one candidate `32 x 32` tile of `C`. Unlike the CUDA implementation, a program instance should not be interpreted as a single CUDA thread; it represents a block of elementwise and matrix operations that Triton maps onto the hardware.

#### Offsets, strides, and row-major addresses

The row offsets and column offsets are constructed as:

```text
a_m_offset = PID_M * 32 + [0, 1, ..., 31]
b_k_offset = PID_K * 32 + [0, 1, ..., 31]
```

The source passes row-major strides explicitly:

```text
A: (stride_am, stride_an) = (N, 1)
B: (stride_bn, stride_bk) = (K, 1)
C: (stride_cm, stride_ck) = (K, 1)
```

For a reduction tile with offsets `a_n_offset`, broadcasting forms a complete matrix of pointers:

```text
a + a_m_offset[:, None] * N + a_n_offset[None, :]
```

The left term has conceptual shape `32 x 1`, the right term has shape `1 x 32`, and broadcasting produces the addresses of a `32 x 32` tile from `A`. The same construction produces a `32 x 32` tile from `B`:

```text
b + b_n_offset[:, None] * K + b_k_offset[None, :]
```

This pointer arithmetic is the Triton equivalent of the row-major indexing performed manually in CUDA. Because the strides are hard-coded from `N` and `K` instead of read from the input tensors, this implementation assumes contiguous row-major tensors, which matches the challenge interface and the tests.

#### Iterating over the reduction dimension

`MAX_N = tl.cdiv(N, BLOCK_SIZE_N)` is the number of 32-element reduction tiles. During every iteration:

1. The program builds a `32 x 32` tile of pointers into `A`.
2. It builds the matching `32 x 32` tile of pointers into `B`.
3. It loads both tiles with masks.
4. It multiplies them with `tl.dot`.
5. It adds the partial product to a float32 accumulator.

The accumulator has shape `32 x 32`:

```text
accumulated_block = tl.zeros((32, 32), dtype=tl.float32)
```

The call

```text
tl.dot(block_a_mn, block_b_nk, accumulated_block, allow_tf32=False)
```

performs the tile matrix product and carries the accumulated result into the next reduction iteration. `allow_tf32=False` requests the non-TF32 path for float32 inputs, keeping the numerical behavior closer to ordinary float32 multiplication than an explicitly TF32-enabled implementation.

#### Masking partial tiles

The matrix dimensions are not required to be multiples of 32. Triton therefore constructs Boolean masks for both input tiles:

```text
(a_m_offset < M) and (a_n_offset < N)
(b_n_offset < N) and (b_k_offset < K)
```

Masked `tl.load` operations use `other=0.0`, so invalid rows, columns, and reduction positions behave as zero padding and do not affect the accumulated result. The final output mask:

```text
(a_m_offset < M) and (b_k_offset < K)
```

prevents the last program instances from storing outside `C`.

#### Why no explicit shared memory or synchronization appears

The algorithm is still tiled even though the Python source contains no `__shared__` array or `__syncthreads()`. The source expresses tiles through block-shaped loads and `tl.dot`; Triton's compiler chooses the low-level data movement and synchronization needed to execute them. Ownership is also simple: every program instance writes a distinct `32 x 32` region of `C`, so program instances do not need atomics or cross-program communication.

This implementation is intentionally fixed rather than auto-tuned. It does not use grouped program ordering, multiple candidate block configurations, explicit `num_warps`, or other advanced scheduling choices. Its behavior follows directly from the three 32-element block sizes present in the source.

### Solution

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    PID_M = tl.program_id(0)
    PID_K = tl.program_id(1)
    MAX_N = tl.cdiv(N, BLOCK_SIZE_N)

    accumulated_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    start_a_m = PID_M * BLOCK_SIZE_M
    a_m_offset = start_a_m + tl.arange(0, BLOCK_SIZE_M)

    start_b_k = PID_K * BLOCK_SIZE_K
    b_k_offset = start_b_k + tl.arange(0, BLOCK_SIZE_K)

    for n in tl.range(MAX_N):
        start_a_n = n * BLOCK_SIZE_N
        a_n_offset = start_a_n + tl.arange(0, BLOCK_SIZE_N)

        a_mn_mask = (a_m_offset[:, None] < M) & (a_n_offset[None, :] < N)
        a_mn_ptrs = a + a_m_offset[:, None] * stride_am + a_n_offset[None, :] * stride_an
        block_a_mn = tl.load(a_mn_ptrs, mask=a_mn_mask, other=0.0)

        start_b_n = n * BLOCK_SIZE_N
        b_n_offset = start_b_n + tl.arange(0, BLOCK_SIZE_N)


        b_nk_mask = (b_n_offset[:, None] < N) & (b_k_offset[None, :] < K)
        b_nk_ptrs = b + b_n_offset[:, None] * stride_bn + b_k_offset[None, :] * stride_bk

        block_b_nk = tl.load(b_nk_ptrs, mask=b_nk_mask, other=0.0)

        accumulated_block = tl.dot(block_a_mn, block_b_nk, accumulated_block, allow_tf32=False)
        # block_ab = tl.dot(block_a_mn, block_b_nk)
        # accumulated_block += block_ab

    c_mk_ptrs = c + a_m_offset[:, None] * stride_cm + b_k_offset[None, :] * stride_ck
    c_mk_mask = (a_m_offset[:, None] < M) & (b_k_offset[None, :] < K)
    tl.store(c_mk_ptrs, accumulated_block, mask=c_mk_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
    )
```

### Test Code

```python
import importlib.util
import sys
from pathlib import Path

import torch


def load_implementation():
    source_path = Path(__file__).resolve().parents[2] / "src" / "triton" / "02_Matrix_Multiplication.py"
    spec = importlib.util.spec_from_file_location("matrix_multiplication_triton_minimal", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    matrix_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    matrix_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device="cuda")
    actual = torch.empty_like(expected)

    load_implementation().solve(matrix_a, matrix_b, actual, 2, 2, 2)
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
| NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | M=8192, N=6144, K=4096 | 3 | 68.536 ms | 63.085 ms | 77.552 ms | 6016.063 GFLOPS |
| NVIDIA GeForce RTX 4090 | PASS | M=8192, N=6144, K=4096 | 3 | 15.875 ms | 15.735 ms | 16.154 ms | 25972.196 GFLOPS |


## PyTorch

### Approach

The PyTorch implementation is intentionally short because the framework already provides matrix multiplication as a built-in tensor operation:

```text
torch.matmul(A, B, out=C)
```

For this challenge, `A` and `B` are two-dimensional tensors with shapes `M x N` and `N x K`. `torch.matmul` checks that the inner dimensions are compatible, computes the `M x K` matrix product, and writes it directly into the supplied output tensor `C`. Using `out=C` is important because the challenge requires the caller-provided output tensor to contain the final result.

PyTorch offers several multiplication interfaces, but they do not all mean the same thing:

- `A * B` performs elementwise multiplication and is not matrix multiplication.
- `A @ B` is Python operator syntax for matrix multiplication and follows `torch.matmul` semantics.
- `torch.mm(A, B)` is specialized for two-dimensional matrix inputs.
- `torch.bmm(A, B)` handles batches of three-dimensional matrices without general broadcasting.
- `torch.matmul(A, B)` supports vectors, matrices, higher-dimensional batches, and broadcasting.
- `torch.einsum(...)` expresses contractions explicitly through index notation and is useful for more complex tensor relationships.

Both `torch.mm` and `A @ B` could express the two-dimensional mathematics of this problem, but the actual source uses `torch.matmul` with an `out` argument, so that is the operation documented and tested here.

There is no explicit grid, thread index, tile size, mask, or synchronization in this implementation. PyTorch dispatches the operation according to the tensors' device and dtype, and its CUDA backend selects the underlying GPU implementation. The arguments `M`, `N`, and `K` remain in the required `solve` signature but are not read directly; the tensor shapes determine the executed matrix multiplication.

### Solution

```python
import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    torch.matmul(A,B,out=C)
```

### Test Code

```python
import importlib.util
import sys
from pathlib import Path

import torch


def load_implementation():
    source_path = Path(__file__).resolve().parents[2] / "src" / "pytorch" / "02_Matrix_Multiplication.py"
    spec = importlib.util.spec_from_file_location("matrix_multiplication_pytorch_minimal", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    matrix_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    matrix_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device="cuda")
    actual = torch.empty_like(expected)

    load_implementation().solve(matrix_a, matrix_b, actual, 2, 2, 2)
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
| NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | M=8192, N=6144, K=4096 | 3 | 26.839 ms | 25.407 ms | 28.341 ms | 15362.601 GFLOPS |
| NVIDIA GeForce RTX 4090 | PASS | M=8192, N=6144, K=4096 | 3 | 7.043 ms | 7.037 ms | 7.053 ms | 58542.190 GFLOPS |

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/matrix-multiplication)
- [Du Ziyuan: CUDA Learning Journey [11] — A Detailed Explanation of Matrix Multiplication](https://dlog.com.cn/posts/cuda11/matmul/)

## Acknowledgements

Special thanks to [Du Ziyuan](https://dlog.com.cn/) for his detailed article on CUDA matrix multiplication. Its clear explanations, diagrams, and animations provided valuable guidance for this document.

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
