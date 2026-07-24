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
