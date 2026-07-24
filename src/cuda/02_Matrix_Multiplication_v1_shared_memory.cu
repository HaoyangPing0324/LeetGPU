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
