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
