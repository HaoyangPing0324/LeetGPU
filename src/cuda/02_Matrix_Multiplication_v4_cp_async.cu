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
