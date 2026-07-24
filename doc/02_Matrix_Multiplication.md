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

## CUDA

### Approach

The challenge multiplies `A[M, N]` by `B[N, K]` and stores `C[M, K]`. One output element is the inner product of a row from `A` and a column from `B`:

\[
C[\text{row},\text{col}]
=
\sum_{\text{inner}=0}^{N-1}
A[\text{row},\text{inner}]
\times
B[\text{inner},\text{col}].
\]

Because the matrices are row-major, the three linear addresses are:

```text
A[row, inner] -> A[row * N + inner]
B[inner, col] -> B[inner * K + col]
C[row, col]   -> C[row * K + col]
```

Every `(row, col)` output coordinate is independent. CUDA therefore parallelizes the two output dimensions, while the reduction over `N` remains inside a thread or a thread-owned output tile.

A conventional CPU implementation expresses this as three nested loops: two loops enumerate the \(M \times K\) output matrix, and the innermost loop performs the length-\(N\) reduction. The arithmetic complexity is \(O(MNK)\). GPU optimization does not change that arithmetic count; it changes where the loops execute, how many outputs are computed together, and how often values are fetched from each level of the memory hierarchy.

![Inner-product view of matrix multiplication](../assets/02_Matrix_Multiplication/image_01.webp)

*One output element combines one row of `A` with one column of `B`.*

<video controls width="760">
  <source src="../assets/02_Matrix_Multiplication/video_01.mp4" type="video/mp4">
  Your Markdown viewer does not support embedded video. Open [the output-element animation](../assets/02_Matrix_Multiplication/video_01.mp4) directly.
</video>

The CUDA files form an optimization sequence. Every version preserves the same `solve(...)` interface and mathematical result. The versions differ in how much output work is assigned to each thread and how data moves through global memory, shared memory, and registers.

The progression follows two related goals:

1. **Increase reuse.** Values first read from global memory should contribute to several output elements before being discarded. Shared-memory tiling enables reuse across a block, while thread tiling enables further reuse inside registers.
2. **Overlap or accelerate movement and computation.** Vectorized transfers reduce load/store instruction overhead, software and hardware pipelines overlap adjacent reduction tiles, and Tensor Cores replace scalar FP32 multiply-add instructions when reduced TF32 input precision is acceptable.

| Version | Main change | Work owned by one thread or warp |
|---|---|---|
| V0 | Direct global-memory dot products | One output element per thread |
| V1 shared | Cooperative shared-memory tiling | One output element per thread |
| V1 1D | Register reuse along the row dimension | `16 x 1` outputs per thread |
| V1 2D | Register-level outer products | `8 x 4` outputs per thread |
| V2 | `float4` cooperative transfers | `8 x 4` outputs per thread |
| V3 | Register prefetch plus two shared-memory buffers | `8 x 4` outputs per thread |
| V4 | Hardware-asynchronous `cp.async` transfers | `8 x 4` outputs per thread |
| V5 | WMMA TF32 Tensor Core operations | One `16 x 16` output tile per warp |

### V0: Naive Global-Memory Kernel

#### Approach

V0 launches a two-dimensional grid with `16 x 16` threads per block. Each thread computes its output coordinate as:

```text
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

The grid uses ceiling division in both output dimensions. A thread first checks `row < M && col < K`; a valid thread then computes one complete dot product and writes one element of `C`.

This mapping replaces the two outer CPU loops with a two-dimensional CUDA grid, but each thread still performs the complete inner reduction serially. It is direct and easy to verify.

Its weakness is data movement. Two threads computing `C[row, col0]` and `C[row, col1]` traverse the same row of `A`, but both independently reload it. Likewise, threads computing different rows for the same output column repeatedly request values from the same column of `B`. Hardware caches may capture part of this reuse, but the kernel does not explicitly preserve the values on chip. V0 therefore exposes abundant output parallelism without improving the locality of the underlying dot products.

#### Solution

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
}
```

### V1: Original `16 x 16` Shared-Memory Tiling

#### Approach

The original implementation keeps one output element per thread but divides the reduction dimension into tiles of 16. One `16 x 16` block produces one `16 x 16` tile of `C`. This introduces an explicit data path:

```text
global memory -> shared memory -> thread registers -> C
```

Global memory holds the full matrices. Shared memory acts as a block-local staging area for the current pair of input tiles. Each thread finally holds its scalar accumulator in a register.

For reduction tile `tile`, thread `(ty, tx)` loads:

```text
A[row, tile * 16 + tx] -> A_shared[ty][tx]
B[tile * 16 + ty, col] -> B_shared[ty][tx]
```

After all 256 threads finish loading, each `A` value is shared by the 16 threads computing different output columns, and each `B` value is shared by the 16 threads computing different output rows. Every thread then performs 16 multiply-add operations:

```text
S += A_shared[ty][inner] * B_shared[inner][tx]
```

Thus, an input value fetched once from global memory can participate in as many as 16 multiply-add operations within the block. For a full reduction tile, the block loads \(16 \times 16\) values from each input and uses them to perform \(16 \times 16 \times 16\) multiply-add operations. The arithmetic is identical to V0, but much of the repeated global-memory traffic is replaced by lower-latency shared-memory access.

<video controls width="760">
  <source src="../assets/02_Matrix_Multiplication/video_02.mp4" type="video/mp4">
  Your Markdown viewer does not support embedded video. Open [the shared-memory tiling animation](../assets/02_Matrix_Multiplication/video_02.mp4) directly.
</video>

Two barriers are necessary for every reduction tile. The first prevents computation from starting before the cooperative load is complete. The second prevents the next tile from overwriting shared memory before every thread has finished using the current tile.

The last reduction tile may be incomplete. Invalid input coordinates are written to shared memory as zero rather than causing an early return. This zero padding contributes nothing to the dot product and lets every thread reach the same barriers, avoiding divergent synchronization. The output store is guarded separately by `row < M && col < K`.

#### Solution

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
}
```

### V1: One-Dimensional Thread Tiling

#### Approach

The 1D Thread Tiling version uses:

```text
BM = 64
BN = 64
BK = 16
TM = 16
```

One block produces a `64 x 64` output tile. Its dimensions are `(64, 4)`, giving 256 threads. A thread owns one output column and 16 output rows:

```text
row = blockIdx.y * 64 + ty * 16 + local_row
col = blockIdx.x * 64 + tx
```

The 16 partial sums remain in `float sum[16]`. During one reduction step, a thread loads one `B` value from shared memory and reuses it across all 16 accumulators:

\[
\text{sum}[r] \mathrel{+}= SA[ty \times TM+r,\text{inner}] \times b,
\quad 0 \le r < 16.
\]

The one-output-per-thread kernel loads one `A` and one `B` shared-memory value to perform one multiply-add at each reduction position. Here the same `B` register feeds 16 multiply-adds. The additional accumulators consume registers, but they increase arithmetic work per shared-memory access and amortize block-level synchronization over more work per thread.

The block still cooperatively fills `SA[64][16]` and `SB[16][64]`. Flattening the thread index makes loading independent of the output ownership mapping. Bounds are checked during cooperative loads and final stores, so partial tiles remain correct.

#### Solution

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
}
```

### V1: Two-Dimensional Thread Tiling

#### Approach

The 2D Thread Tiling version uses:

```text
BM = 64
BN = 64
BK = 16
TM = 8
TN = 4
```

The block dimensions are `(16, 8)`, or 128 threads. Each thread owns an `8 x 4` micro-tile containing 32 output elements:

```text
row = blockIdx.y * 64 + ty * 8 + local_row
col = blockIdx.x * 64 + tx * 4 + local_col
```

For one reduction position, the thread loads eight `A` values and four `B` values from shared memory into registers. Their outer product updates all 32 accumulators:

\[
C_{\text{micro}}
\mathrel{+}=
\begin{bmatrix}
a_0\\a_1\\\vdots\\a_7
\end{bmatrix}
\begin{bmatrix}
b_0&b_1&b_2&b_3
\end{bmatrix}.
\]

This is another view of matrix multiplication. A scalar output is naturally described as an inner product, but an output tile can be accumulated as a sequence of rank-one outer products over the reduction dimension. At one reduction position, the eight-element `A` column fragment and the four-element `B` row fragment update the entire thread-owned `8 x 4` output tile.

Each `A` register contributes to four output columns, and each `B` register contributes to eight output rows. The thread therefore performs 32 multiply-add operations from only 12 shared-memory scalar loads.

This is the central benefit of two-dimensional thread tiling: the mathematical work is unchanged, but both operands are reused in registers before another shared-memory access is required.

#### Solution

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
}
```

### V2: Vectorized `float4` Transfers (`BK = 32`)

#### Approach

V2 retains the `8 x 4` register micro-tile and changes the reduction tile to `BK = 32`. A `float4` contains four adjacent FP32 values and occupies 16 bytes. The kernel uses it for four adjacent values at a time in:

- global-memory loads from `A` and `B`;
- stores into shared memory;
- shared-memory reads of adjacent `B` values;
- final stores to adjacent columns of `C`.

A vector load is valid only when its address is suitably aligned and all four elements exist. The vectorized kernel is therefore selected only when `N` and `K` are divisible by four. When either condition is false, `solve(...)` launches the scalar safe implementation. This fallback is necessary for the general LeetGPU interface; vectorization must not trade away correctness on irregular shapes.

Within the optimized path, output rows in the final block are still guarded independently. Because `K` is a multiple of four, a valid four-column group is either fully inside the matrix or fully outside it.

The row-major layout makes consecutive columns contiguous. It is therefore natural to vectorize along `A`'s reduction coordinate, along `B`'s output-column coordinate, and along `C`'s output-column coordinate. One vector instruction replaces four scalar instructions at these transfer sites. The number of bytes and the required \(2MNK\) floating-point operations do not change; vectorization primarily reduces instruction and address-generation overhead. Its benefit consequently depends on alignment, instruction issue, and the balance between memory movement and arithmetic rather than following automatically from the wider type.

#### Solution

```cuda
#include <cuda_runtime.h>

#ifndef MATMUL_V2_BM
#define MATMUL_V2_BM 64
#endif
#ifndef MATMUL_V2_BN
#define MATMUL_V2_BN 64
#endif
#ifndef MATMUL_V2_BK
#define MATMUL_V2_BK 32
#endif
#ifndef MATMUL_V2_TM
#define MATMUL_V2_TM 8
#endif
#ifndef MATMUL_V2_TN
#define MATMUL_V2_TN 4
#endif

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

template <int BM = MATMUL_V2_BM, int BN = MATMUL_V2_BN,
          int BK = MATMUL_V2_BK, int TM = MATMUL_V2_TM,
          int TN = MATMUL_V2_TN>
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
        constexpr int BM = MATMUL_V2_BM, BN = MATMUL_V2_BN;
        constexpr int BK = MATMUL_V2_BK, TM = MATMUL_V2_TM;
        constexpr int TN = MATMUL_V2_TN;
        const dim3 block(BN / TN, BM / TM);
        const dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
        matrix_multiplication_v2<BM, BN, BK, TM, TN>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
}
```

### V2 Control: Vectorized Single Buffer (`BK = 16`)

#### Approach

This control version compiles the same single-buffered V2 kernel with `BK = 16`, matching the reduction-tile depth used by V3. All other V2 work ownership, vectorized transfers, register micro-tile, boundary fallback, and launch geometry remain unchanged.

The matched configuration separates the effect of reduction-tile depth from the effect of double buffering. On both tested GPUs, the single-buffered `BK = 16` control remains faster than V3. Therefore, V3's lower measured performance cannot be explained only by comparing V2's original `BK = 32` against V3's `BK = 16`; the additional prefetch registers, two shared-memory buffers, buffer switching, and synchronization also fail to repay their cost in this implementation.

#### Solution

```cuda
#define MATMUL_V2_BK 16

#include "02_Matrix_Multiplication_v2_vectorized.cu"
```

### V2 Control: Vectorized Single Buffer (`128 x 128`)

#### Approach

This control keeps the V2 single-buffered algorithm and changes only the compile-time tile parameters to `BM = 128`, `BN = 128`, and `BK = 16`. The per-thread `8 x 4` micro-tile, `float4` transfers, scalar boundary fallback, and accumulation logic remain unchanged.

The block grows from 128 to 512 threads and produces four times as many output elements, but the larger tile does not improve this single-buffered kernel. It is slower than the matched `64 x 64`, `BK = 16` control on both tested GPUs. The extra block-level reuse is therefore insufficient by itself to offset the larger block's scheduling and resource costs.

#### Solution

```cuda
#define MATMUL_V2_BM 128
#define MATMUL_V2_BN 128
#define MATMUL_V2_BK 16
#define MATMUL_V2_TM 8
#define MATMUL_V2_TN 4

#include "02_Matrix_Multiplication_v2_vectorized.cu"
```

### V3: Register Prefetch and Double-Buffered Shared Memory

#### Base `64 x 64` Configuration

##### Approach

V3 uses:

```text
BM = 64
BN = 64
BK = 16
TM = 8
TN = 4
```

It allocates two copies of each shared-memory tile:

```text
SA[2][64][16]
SB[2][16][64]
```

The first tile is loaded before the main loop. This is the **pipeline prologue**: no computation can begin until one complete input-tile pair is available in shared memory.

For each later tile, a thread first issues global loads into private prefetch registers. It then computes from the current shared-memory buffer, waits until all threads have finished consuming that buffer, writes the prefetched values into the alternate buffer, and finally synchronizes before exchanging the read and write buffer indices. This is the **steady state**, in which work for reduction tile \(i\) is interleaved with data movement for tile \(i+1\).

After the last prefetched tile becomes current, the kernel computes it without requesting another tile. This is the **pipeline epilogue**.

The intended pipeline is:

```text
tile i + 1: global memory -> prefetch registers
tile i:     shared memory -> accumulator registers
tile i + 1: prefetch registers -> alternate shared-memory buffer
```

This is software double buffering with register prefetch, not an asynchronous `cp.async` pipeline. The global loads are issued before the current tile is consumed, giving the compiler and GPU an opportunity to overlap outstanding memory operations with independent arithmetic. However, the implementation still needs block-wide barriers when a shared buffer changes roles.

The actual source keeps `SA[buffer][row][inner]` and `SB[buffer][inner][col]` in straightforward row-major tile layouts. It does not silently add a transposed shared-memory layout or another unimplemented access transformation. This detail matters because a pipeline cannot be evaluated independently of its real load pattern, register footprint, shared-memory layout, and synchronization schedule.

Double buffering is not automatically faster. It doubles shared-memory storage and adds prefetch registers, control flow, a pipeline prologue, and an epilogue. Those costs can reduce occupancy or offset hidden latency. In the measured workload, the base `64 x 64` V3 is slightly slower than the matched single-buffer V2 control on both tested GPUs. The later `128 x 128` V3 configuration is faster because its larger output tile changes reuse and scheduling at the same time; the separate controls are retained so that this result is not incorrectly attributed to double buffering alone.

##### Solution

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
}
#endif
```

#### Large-Tile `128 x 128` Configuration

##### Approach

The large-tile configuration changes V3's compile-time parameters to:

```text
BM = 128
BN = 128
BK = 16
TM = 8
TN = 4
```

The block contains `(128 / 4) x (128 / 8) = 32 x 16 = 512` threads. Each thread still owns 32 output elements, while the block collectively produces a `128 x 128` tile. Its double-buffered input storage occupies 32 KiB:

```text
2 * (128 * 16 + 16 * 128) * sizeof(float)
```

The larger output tile increases block-level reuse and amortizes input-tile loads across more output elements, while retaining exactly the same register-prefetch and double-buffer mechanism as the base V3 configuration. It also raises the block thread count and changes register demand, occupancy, and scheduling, so it is a parameter configuration of V3 rather than a separate optimization principle.

Under the revised benchmark, this configuration is the fastest project CUDA implementation on both tested GPUs. However, the matched single-buffered experiment becomes slower when enlarged to the same `128 x 128` tile. The gain therefore does not come from tile size alone: it appears only when the larger output tile is combined with this double-buffered data path. The measurement establishes an interaction between the two choices, not a universal rule that larger tiles are faster.

##### Solution

```cuda
#define MATMUL_BM 128
#define MATMUL_BN 128
#define MATMUL_BK 16
#define MATMUL_TM 8
#define MATMUL_TN 4

#include "02_Matrix_Multiplication_v3_double_buffered.cu"
```

#### Controlled Comparison

All four rows below use `BK = 16`, an `8 x 4` per-thread micro-tile, deterministic nonzero inputs, five warm-up iterations, and ten measured iterations.

| Buffering | Output Tile | RTX 5070 Ti Laptop GPU | RTX 4090 |
|---|---:|---:|---:|
| Single buffer | `64 x 64` | 38.761 ms | 11.314 ms |
| Single buffer | `128 x 128` | 38.842 ms | 13.146 ms |
| Double buffer | `64 x 64` | 39.642 ms | 11.514 ms |
| Double buffer | `128 x 128` | **34.590 ms** | **10.052 ms** |

At `64 x 64`, double buffering has only a small and platform-dependent effect: it is slightly faster on the RTX 4090 and slightly slower on the RTX 5070 Ti Laptop GPU. Enlarging the single-buffered kernel to `128 x 128` is slower on both GPUs. In contrast, the `128 x 128` double-buffered configuration is substantially faster than the single-buffered large tile and is the fastest common configuration in the table. The measured acceleration therefore comes from the interaction between the larger output tile and the double-buffered data path, not from a universal rule that either choice is independently faster.

The following compiler resource report was produced with `nvcc -O2 -arch=sm_89 -Xptxas=-v` for the RTX 4090 target:

| Buffering | Output Tile | Threads per Block | Registers per Thread | Shared Memory per Block | Spills |
|---|---:|---:|---:|---:|---:|
| Single buffer | `64 x 64` | 128 | 86 | 8 KiB | 0 |
| Single buffer | `128 x 128` | 512 | 86 | 16 KiB | 0 |
| Double buffer | `64 x 64` | 128 | 122 | 16 KiB | 0 |
| Double buffer | `128 x 128` | 512 | 100 | 32 KiB | 0 |

The report confirms the resource trade-off. The single-buffered large tile keeps the same 86 registers per thread but quadruples the thread count, so its register allocation per block grows from 11,008 to 44,032 registers. The base double-buffered kernel raises register use to 122 per thread because of prefetch state. For the large double-buffered instantiation, the compiler uses 100 registers per thread and no spills; the block performs four times as much output work while preserving the same per-thread micro-tile. These facts explain why occupancy and reuse can change sharply with the configuration, although register and shared-memory counts alone do not prove the exact runtime stall behavior.

### V4: Hardware-Asynchronous `cp.async` Pipeline

#### Approach

V4 keeps the `128 x 128` output tile, `BK = 16`, and `8 x 4` per-thread micro-tile from the large V3 configuration, but changes how the next input tile reaches shared memory.

V3 first loads global-memory values into ordinary thread registers and later stores those registers into the alternate shared-memory buffer. V4 instead issues the Ampere-or-newer PTX instruction:

```text
cp.async.cg.shared.global
```

Each instruction copies 16 bytes directly from global memory to shared memory. `cp.async.commit_group` publishes the copies issued by a thread, and `cp.async.wait_group 0` waits until all outstanding groups needed by the block are complete.

The pipeline has a prologue, steady state, and epilogue:

1. The prologue asynchronously fills shared buffer 0 and waits before its first use.
2. During a steady-state iteration, the block issues copies for tile `i + 1` into the alternate buffer before computing tile `i`.
3. The arithmetic loop reads only the current buffer, so the GPU can overlap its fused multiply-add work with the outstanding global-to-shared copies.
4. Before the buffers exchange roles, `cp.async.wait_group 0` guarantees completion and `__syncthreads()` makes the new tile visible to every thread.
5. The final tile has no successor, so the loop finishes without issuing another copy group.

Out-of-range 16-byte vectors use the source-size operand with zero valid bytes. The hardware zero-fills the destination shared-memory vector without dereferencing an invalid global address. Inputs whose reduction or output-column dimensions are not compatible with four-float vector transfers use the scalar fallback. Devices below compute capability 8.0 also use the fallback because they do not provide `cp.async`.

This is a genuine hardware-asynchronous pipeline, unlike V3's register-prefetch software pipeline. It is nevertheless not guaranteed to be faster: both versions still require synchronization when a shared buffer changes roles, and the `512`-thread block, address generation, copy-group management, and arithmetic instruction stream all compete for execution resources.

#### Solution

```cuda
#include <cuda_runtime.h>

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 4;
constexpr int THREADS_X = BN / TN;
constexpr int THREADS_Y = BM / TM;
constexpr int THREAD_COUNT = THREADS_X * THREADS_Y;

__global__ void scalar_fallback_v4(const float* A, const float* B, float* C,
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_16(void* shared_destination,
                                            const void* global_source,
                                            int valid_bytes) {
    const unsigned int shared_address =
        static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(shared_address), "l"(global_source), "r"(valid_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" : :);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" : :);
}
#endif

__global__ void matrix_multiplication_v4_cp_async(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) float shared_a[2][BM][BK];
    __shared__ __align__(16) float shared_b[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    float accumulators[TM][TN] = {{0.0f}};

    auto issue_tile = [&](int tile, int buffer) {
        constexpr int A_VECTORS = BM * BK / 4;
        constexpr int B_VECTORS = BK * BN / 4;
        const int inner_base = tile * BK;

        for (int vector_index = tid;
             vector_index < A_VECTORS;
             vector_index += THREAD_COUNT) {
            const int element_index = vector_index * 4;
            const int local_row = element_index / BK;
            const int local_inner = element_index % BK;
            const int row = blockIdx.y * BM + local_row;
            const int inner = inner_base + local_inner;
            const int valid_bytes = (row < M && inner + 3 < N) ? 16 : 0;
            const float* source = A + static_cast<size_t>(row < M ? row : 0) * N +
                                  (inner + 3 < N ? inner : 0);
            cp_async_16(&shared_a[buffer][local_row][local_inner],
                        source, valid_bytes);
        }

        for (int vector_index = tid;
             vector_index < B_VECTORS;
             vector_index += THREAD_COUNT) {
            const int element_index = vector_index * 4;
            const int local_inner = element_index / BN;
            const int local_col = element_index % BN;
            const int inner = inner_base + local_inner;
            const int col = blockIdx.x * BN + local_col;
            const int valid_bytes = (inner < N && col + 3 < K) ? 16 : 0;
            const float* source = B + static_cast<size_t>(inner < N ? inner : 0) * K +
                                  (col + 3 < K ? col : 0);
            cp_async_16(&shared_b[buffer][local_inner][local_col],
                        source, valid_bytes);
        }
        cp_async_commit();
    };

    const int tile_count = (N + BK - 1) / BK;
    issue_tile(0, 0);
    cp_async_wait_all();
    __syncthreads();

    int read_buffer = 0;
    for (int tile = 0; tile < tile_count; ++tile) {
        const bool has_next_tile = tile + 1 < tile_count;
        if (has_next_tile) {
            issue_tile(tile + 1, read_buffer ^ 1);
        }

        #pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
            const float4 b = *reinterpret_cast<const float4*>(
                &shared_b[read_buffer][inner][tx * TN]);
            #pragma unroll
            for (int local_row = 0; local_row < TM; ++local_row) {
                const float a =
                    shared_a[read_buffer][ty * TM + local_row][inner];
                accumulators[local_row][0] += a * b.x;
                accumulators[local_row][1] += a * b.y;
                accumulators[local_row][2] += a * b.z;
                accumulators[local_row][3] += a * b.w;
            }
        }

        if (has_next_tile) {
            cp_async_wait_all();
            __syncthreads();
            read_buffer ^= 1;
        }
    }

    #pragma unroll
    for (int local_row = 0; local_row < TM; ++local_row) {
        const int row = blockIdx.y * BM + ty * TM + local_row;
        const int col = blockIdx.x * BN + tx * TN;
        if (row < M && col + 3 < K) {
            *reinterpret_cast<float4*>(&C[static_cast<size_t>(row) * K + col]) =
                make_float4(accumulators[local_row][0],
                            accumulators[local_row][1],
                            accumulators[local_row][2],
                            accumulators[local_row][3]);
        }
    }
#endif
}

}  // namespace

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    static const bool supports_cp_async = [] {
        int device = 0;
        cudaDeviceProp properties{};
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&properties, device);
        return properties.major >= 8;
    }();

    if (!supports_cp_async || (N & 3) != 0 || (K & 3) != 0) {
        const dim3 block(16, 16);
        const dim3 grid((K + 15) / 16, (M + 15) / 16);
        scalar_fallback_v4<<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    const dim3 block(THREADS_X, THREADS_Y);
    const dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    matrix_multiplication_v4_cp_async<<<grid, block>>>(A, B, C, M, N, K);
}
```

### V5: WMMA TF32 Tensor Core Kernel

#### Approach

V5 changes both the execution unit and numerical format. The public interface still receives and returns FP32 matrices, but the fast path explicitly rounds input values to TensorFloat-32 and performs the matrix multiply-accumulate operations on Tensor Cores with FP32 accumulators.

TF32 keeps the eight-bit exponent range of FP32 but uses a ten-bit explicit mantissa. The conversion is performed with:

```text
cvt.rna.tf32.f32
```

This means V5 is not numerically identical to the strict FP32 CUDA Core versions. It trades input mantissa precision for access to Tensor Core matrix instructions. The test therefore uses a `5e-3` scaled tolerance while still rejecting every non-finite result and comparing every output element with the CPU reference.

One block produces a `128 x 128` output tile and reduces `K` in chunks of 32. The block contains 16 warps. Each warp owns one `16 x 64` strip of the output, represented by four `16 x 16` accumulator fragments. Two warps cover the left and right halves of the same 16-row strip.

For each `K=32` shared-memory tile:

1. All 512 threads cooperatively load `A[128, 32]` and `B[32, 128]`, converting each value to TF32.
2. A single block synchronization makes both tiles visible.
3. The tile is divided into four WMMA reduction steps because TF32 WMMA uses `16 x 16 x 8` fragments.
4. Each warp loads one `A` fragment and four `B` fragments per step.
5. `wmma::mma_sync` updates the four FP32 accumulator fragments.
6. After all reduction tiles, `wmma::store_matrix_sync` writes the fragments to row-major FP32 output memory.

The fast path requires `M` and `K` to align with the `128 x 128` output tile and `N` to align with `BK=32`. Other shapes, older architectures, and boundary cases use the scalar FP32 fallback. This preserves the challenge interface and general correctness while keeping the WMMA kernel free of partial-fragment stores.

Tensor Cores provide high peak throughput, but this implementation also pays for explicit TF32 conversion, shared-memory staging, frequent fragment loads, block synchronization, and WMMA fragment ownership. The measurements therefore demonstrate a correct low-level WMMA implementation, not an automatic claim that it beats the carefully tuned CUDA Core kernel.

#### Solution

```cuda
#include <cuda_runtime.h>
#include <mma.h>

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_PER_BLOCK = 16;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int WMMA_K = 8;
constexpr int OUTPUT_FRAGMENTS_PER_WARP = BN / 32;

__device__ __forceinline__ float to_tf32(float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned int tf32_bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(tf32_bits) : "f"(value));
    return __uint_as_float(tf32_bits);
#else
    return value;
#endif
}

__global__ void scalar_fallback_v5(const float* A, const float* B, float* C,
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

__global__ void matrix_multiplication_v5_wmma_tf32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;

    __shared__ __align__(16) float shared_a[BM][BK];
    __shared__ __align__(16) float shared_b[BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2;
    const int warp_col_group = warp_id % 2;

    wmma::fragment<wmma::accumulator, 16, 16, WMMA_K, float>
        accumulators[OUTPUT_FRAGMENTS_PER_WARP];
    #pragma unroll
    for (int fragment = 0; fragment < OUTPUT_FRAGMENTS_PER_WARP; ++fragment) {
        wmma::fill_fragment(accumulators[fragment], 0.0f);
    }

    for (int inner_base = 0; inner_base < N; inner_base += BK) {
        for (int index = tid; index < BM * BK;
             index += THREADS_PER_BLOCK) {
            const int local_row = index / BK;
            const int local_inner = index % BK;
            const int row = blockIdx.y * BM + local_row;
            const float value = A[static_cast<size_t>(row) * N +
                                  inner_base + local_inner];
            shared_a[local_row][local_inner] = to_tf32(value);
        }
        for (int index = tid; index < BK * BN;
             index += THREADS_PER_BLOCK) {
            const int local_inner = index / BN;
            const int local_col = index % BN;
            const int col = blockIdx.x * BN + local_col;
            const float value = B[static_cast<size_t>(inner_base + local_inner) * K +
                                  col];
            shared_b[local_inner][local_col] = to_tf32(value);
        }
        __syncthreads();

        const int local_row = warp_row * 16;
        #pragma unroll
        for (int inner = 0; inner < BK; inner += WMMA_K) {
            wmma::fragment<wmma::matrix_a, 16, 16, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> a_fragment;
            wmma::load_matrix_sync(
                a_fragment, &shared_a[local_row][inner], BK);

            #pragma unroll
            for (int fragment = 0;
                 fragment < OUTPUT_FRAGMENTS_PER_WARP;
                 ++fragment) {
                wmma::fragment<wmma::matrix_b, 16, 16, WMMA_K,
                               wmma::precision::tf32,
                               wmma::row_major> b_fragment;
                wmma::load_matrix_sync(
                    b_fragment,
                    &shared_b[inner][warp_col_group * 64 + fragment * 16],
                    BN);
                wmma::mma_sync(accumulators[fragment], a_fragment,
                               b_fragment, accumulators[fragment]);
            }
        }
        __syncthreads();
    }

    const int output_row = blockIdx.y * BM + warp_row * 16;
    const int output_col = blockIdx.x * BN + warp_col_group * 64;
    #pragma unroll
    for (int fragment = 0; fragment < OUTPUT_FRAGMENTS_PER_WARP; ++fragment) {
        wmma::store_matrix_sync(
            &C[static_cast<size_t>(output_row) * K +
               output_col + fragment * 16],
            accumulators[fragment], K, wmma::mem_row_major);
    }
#endif
}

}  // namespace

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    static const bool supports_wmma_tf32 = [] {
        int device = 0;
        cudaDeviceProp properties{};
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&properties, device);
        return properties.major >= 8;
    }();

    if (!supports_wmma_tf32 ||
        M % BM != 0 || N % BK != 0 || K % BN != 0) {
        const dim3 block(16, 16);
        const dim3 grid((K + 15) / 16, (M + 15) / 16);
        scalar_fallback_v5<<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid(K / BN, M / BM);
    matrix_multiplication_v5_wmma_tf32<<<grid, block>>>(A, B, C, M, N, K);
}
```

### Test Methodology

All CUDA methods use the same test file and must pass nine correctness shapes, including scalar, rectangular, aligned, non-aligned, partial-tile, and multi-block dimensions. The checker rejects non-finite output values and compares every output element with a CPU reference using a scaled tolerance of `5e-3`. This tolerance covers the deliberate TF32 input rounding in V5; the strict FP32 methods are evaluated by the same test so that every row in the comparison uses one common procedure. The `129 x 96` by `96 x 132` case exercises V4's asynchronous fast path with partial output tiles, while the `256 x 96` by `96 x 256` case exercises multiple WMMA output blocks.

The performance case uses `M=8192`, `N=6144`, and `K=4096`. Inputs are initialized with deterministic nonzero values so that zero-filled allocations do not produce an unrepresentative memory behavior. Every method is compiled for the native remote GPU architecture. Five warm-up iterations run before ten CUDA-event measurements. The tables below report the average, minimum, maximum, and calculated floating-point throughput from those ten measurements.

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

The following tables intentionally keep results from the same GPU together. They compare methods within one platform; they are not intended to rank one GPU against the other. `Default` currently selects the V3 large-tile implementation, so its row is a repeated run of that source path rather than another algorithm.

#### NVIDIA GeForce RTX 5070 Ti Laptop GPU

| Method | Arithmetic | Status | Average Time | Minimum Time | Maximum Time | Performance |
|---|---|:---:|---:|---:|---:|---:|
| V0 naive | FP32 | PASS | 383.370 ms | 348.679 ms | 417.879 ms | 1075.506 GFLOPS |
| V1 shared memory | FP32 | PASS | 250.884 ms | 224.301 ms | 302.463 ms | 1643.454 GFLOPS |
| V1 1D thread tiling | FP32 | PASS | 135.891 ms | 128.005 ms | 150.389 ms | 3034.181 GFLOPS |
| V1 2D thread tiling | FP32 | PASS | 123.423 ms | 117.553 ms | 135.046 ms | 3340.669 GFLOPS |
| V2 vectorized, `BK=32` | FP32 | PASS | 46.376 ms | 44.035 ms | 48.737 ms | 8890.723 GFLOPS |
| V2 vectorized, `BK=16` | FP32 | PASS | 38.761 ms | 36.343 ms | 40.281 ms | 10637.490 GFLOPS |
| V2 vectorized, `128 x 128` | FP32 | PASS | 38.842 ms | 37.423 ms | 41.546 ms | 10615.226 GFLOPS |
| V3 software double buffer, `64 x 64` | FP32 | PASS | 39.642 ms | 34.580 ms | 42.673 ms | 10400.993 GFLOPS |
| V3 software double buffer, `128 x 128` | FP32 | PASS | 34.590 ms | 32.101 ms | 37.409 ms | 11920.091 GFLOPS |
| **V4 hardware `cp.async`** | **FP32** | **PASS** | **34.572 ms** | **31.259 ms** | **37.045 ms** | **11926.402 GFLOPS** |
| V5 WMMA Tensor Core | TF32 input, FP32 accumulation | PASS | 57.805 ms | 52.967 ms | 66.704 ms | 7132.925 GFLOPS |
| Default (V3 `128 x 128`) | FP32 | PASS | 34.486 ms | 31.874 ms | 37.132 ms | 11956.000 GFLOPS |

On this GPU, V4 has the lowest average in the final common test, but its `34.572 ms` result is effectively tied with the matching large V3 result of `34.590 ms` relative to the observed run-to-run spread. Repeated V4 runs ranged from roughly 32 to 35 ms. The evidence therefore shows that hardware-asynchronous copies are competitive on this platform, not that the 0.018 ms final difference is significant. V5 is correct under the documented TF32 tolerance but is slower than the tuned CUDA Core kernels because this hand-written WMMA mapping pays substantial conversion, shared-memory, synchronization, and fragment-management costs.

#### NVIDIA GeForce RTX 4090

| Method | Arithmetic | Status | Average Time | Minimum Time | Maximum Time | Performance |
|---|---|:---:|---:|---:|---:|---:|
| V0 naive | FP32 | PASS | 83.140 ms | 82.831 ms | 83.375 ms | 4959.321 GFLOPS |
| V1 shared memory | FP32 | PASS | 59.379 ms | 58.985 ms | 59.865 ms | 6943.818 GFLOPS |
| V1 1D thread tiling | FP32 | PASS | 19.484 ms | 19.393 ms | 19.505 ms | 21161.508 GFLOPS |
| V1 2D thread tiling | FP32 | PASS | 13.446 ms | 13.420 ms | 13.465 ms | 30665.525 GFLOPS |
| V2 vectorized, `BK=32` | FP32 | PASS | 11.646 ms | 11.098 ms | 12.106 ms | 35404.086 GFLOPS |
| V2 vectorized, `BK=16` | FP32 | PASS | 11.314 ms | 11.213 ms | 11.345 ms | 36441.761 GFLOPS |
| V2 vectorized, `128 x 128` | FP32 | PASS | 13.146 ms | 13.135 ms | 13.173 ms | 31363.839 GFLOPS |
| V3 software double buffer, `64 x 64` | FP32 | PASS | 11.514 ms | 11.380 ms | 11.654 ms | 35809.707 GFLOPS |
| **V3 software double buffer, `128 x 128`** | **FP32** | **PASS** | **10.052 ms** | **9.939 ms** | **10.213 ms** | **41019.869 GFLOPS** |
| V4 hardware `cp.async` | FP32 | PASS | 10.293 ms | 10.230 ms | 10.415 ms | 40056.186 GFLOPS |
| V5 WMMA Tensor Core | TF32 input, FP32 accumulation | PASS | 12.675 ms | 12.107 ms | 13.133 ms | 32528.826 GFLOPS |
| Default (V3 `128 x 128`) | FP32 | PASS | 10.096 ms | 10.010 ms | 10.144 ms | 40840.572 GFLOPS |

On this GPU, the large V3 software pipeline remains the fastest measured implementation. V4 overlaps copies with arithmetic but does not recover the extra copy-group and synchronization overhead. V5 again demonstrates functional Tensor Core execution without outperforming the established FP32 path. The result is a useful boundary: Tensor Core peak throughput alone does not guarantee a faster end-to-end kernel when data preparation and fragment scheduling are not as mature as a production GEMM library.

## Triton

### Approach

The Triton implementation computes the same row-major matrix product as CUDA:

\[
C[m,k] = \sum_{n=0}^{N-1} A[m,n]B[n,k].
\]

A Triton program instance owns one `64 x 64` output tile. Unlike CUDA source code, the implementation does not explicitly assign individual scalar outputs to `threadIdx` coordinates. It describes whole blocks of indices and values; Triton's compiler maps those operations onto GPU threads and warps. The reduction dimension is processed in chunks of 32:

```text
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 8
```

The output grid is flattened to one dimension. `tl.program_id(0)` identifies the current program instance. From that linear ID, the kernel derives `PID_M` and `PID_K`, which identify the output-row and output-column tiles. The complete grid contains:

\[
\left\lceil \frac{M}{64} \right\rceil
\times
\left\lceil \frac{K}{64} \right\rceil
\]

program instances.

Programs are ordered in groups of up to eight M tiles before advancing farther across K. This changes only execution order, not which output tile each program owns. Nearby programs are more likely to reuse cached regions of `A` and `B`, whereas a simple row-major ordering can move farther through one output dimension before returning to reusable input data.

For one program, the row and column offsets are:

```text
a_m_offset = PID_M * 64 + [0, ..., 63]
b_k_offset = PID_K * 64 + [0, ..., 63]
```

During each reduction step, `tl.arange` creates the row, reduction, and column offsets. Adding singleton dimensions with `[:, None]` and `[None, :]` broadcasts those vectors into a `64 x 32` pointer tile from `A` and a `32 x 64` pointer tile from `B`. The row-major strides are passed explicitly as `(N, 1)`, `(K, 1)`, and `(K, 1)` for `A`, `B`, and `C`.

Masks serve the same correctness role as bounds checks and zero padding in the CUDA kernels. A masked input location is loaded as zero, so it contributes nothing to the reduction. A masked output location is not stored. Consequently, the same kernel handles dimensions that are not multiples of 64 or 32 without launching a separate boundary kernel.

`tl.dot` multiplies the two input tiles and accumulates into a float32 `64 x 64` block. `allow_tf32=False` keeps the calculation on the non-TF32 path used by this implementation. After the reduction loop, a final mask prevents stores outside `C`.

The launch uses four warps and three pipeline stages per program. Compared with the previous fixed `32 x 32` output tile and ungrouped two-dimensional grid, the larger tile performs more work per program and the grouped ordering improves locality. Under the common benchmark, this reduces the average time from 16.907 ms to 9.803 ms on the RTX 4090 and from 78.710 ms to 32.518 ms on the RTX 5070 Ti Laptop GPU.

### Solution

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    PID_M = first_pid_m + (pid % num_pid_in_group) % group_size_m
    PID_K = (pid % num_pid_in_group) // group_size_m
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

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_K),)
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
        GROUP_SIZE_M = GROUP_SIZE_M,
        num_warps=4,
        num_stages=3,
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
| NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | M=8192, N=6144, K=4096 | 10 | 32.518 ms | 30.066 ms | 34.157 ms | 12679.673 GFLOPS |
| NVIDIA GeForce RTX 4090 | PASS | M=8192, N=6144, K=4096 | 10 | 9.803 ms | 8.994 ms | 10.005 ms | 42061.461 GFLOPS |


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
| NVIDIA GeForce RTX 5070 Ti Laptop GPU | PASS | M=8192, N=6144, K=4096 | 10 | 29.305 ms | 25.729 ms | 31.689 ms | 14069.752 GFLOPS |
| NVIDIA GeForce RTX 4090 | PASS | M=8192, N=6144, K=4096 | 10 | 7.469 ms | 7.188 ms | 7.559 ms | 55204.880 GFLOPS |

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/matrix-multiplication)
- [Du Ziyuan: CUDA Learning Journey [11] — A Detailed Explanation of Matrix Multiplication](https://dlog.com.cn/posts/cuda11/matmul/)

## Acknowledgements

This document adapts explanatory material, diagrams, and animations from Du Ziyuan's article, [CUDA Learning Journey [11] — A Detailed Explanation of Matrix Multiplication](https://dlog.com.cn/posts/cuda11/matmul/). The original work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The adapted documentation and locally reproduced media are distributed under the same license. Changes include restructuring the explanation around the LeetGPU interface, matching the discussion and code to this project's implementations, and adding project-specific tests and measured results.

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
