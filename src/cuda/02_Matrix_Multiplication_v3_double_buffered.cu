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
