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
