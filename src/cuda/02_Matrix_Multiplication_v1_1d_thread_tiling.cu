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
