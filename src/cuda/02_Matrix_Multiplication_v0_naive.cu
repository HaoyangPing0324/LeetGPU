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
