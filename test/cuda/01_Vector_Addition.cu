#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

extern "C" void solve(const float* a, const float* b, float* c, int n);

static bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << "CUDA error in " << operation << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    bool allocate(size_t count, const char* operation) {
        return cuda_ok(cudaMalloc(&data_, count * sizeof(T)), operation);
    }

    T* get() { return data_; }
    const T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

static bool run_correctness_case(int n) {
    // ==================== Core correctness tests begin ====================
    const size_t count = static_cast<size_t>(n);
    const size_t bytes = count * sizeof(float);
    std::vector<float> a(count);
    std::vector<float> b(count);
    std::vector<float> c(count);

    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>((i * 17) % 101 - 50) / 13.0f;
        b[i] = static_cast<float>((i * 29) % 97 - 48) / 11.0f;
    }

    DeviceBuffer<float> d_a;
    DeviceBuffer<float> d_b;
    DeviceBuffer<float> d_c;
    if (!d_a.allocate(count, "cudaMalloc(d_a)") ||
        !d_b.allocate(count, "cudaMalloc(d_b)") ||
        !d_c.allocate(count, "cudaMalloc(d_c)") ||
        !cuda_ok(cudaMemcpy(d_a.get(), a.data(), bytes, cudaMemcpyHostToDevice), "copy a") ||
        !cuda_ok(cudaMemcpy(d_b.get(), b.data(), bytes, cudaMemcpyHostToDevice), "copy b")) {
        return false;
    }

    solve(d_a.get(), d_b.get(), d_c.get(), n);
    if (!cuda_ok(cudaGetLastError(), "vector_add kernel") ||
        !cuda_ok(cudaDeviceSynchronize(), "vector_add synchronize") ||
        !cuda_ok(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost), "copy c")) {
        return false;
    }

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        const float tolerance = 1e-6f * std::max(1.0f, std::fabs(expected));
        if (!std::isfinite(c[i]) ||
            std::fabs(c[i] - expected) > tolerance) {
            if (errors < 5) {
                std::cerr << "Mismatch at " << i << ": " << c[i]
                          << " != " << expected << '\n';
            }
            ++errors;
        }
    }

    if (errors != 0) {
        std::cerr << "[FAIL] vector_add correctness N=" << n
                  << ", mismatches=" << errors << '\n';
        return false;
    }

    std::cout << "[PASS] vector_add correctness N=" << n << '\n';
    // ==================== Core correctness tests end ====================
    return true;
}

__global__ void fill_buffer(float* values, size_t count, float scale) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const int centered = static_cast<int>(index % 17) - 8;
        values[index] = static_cast<float>(centered) * scale;
    }
}

static bool run_performance_test() {
    // ==================== Core performance test begins ====================
    constexpr int n = 25000000;
    constexpr int warmup_iterations = 5;
    constexpr int measured_iterations = 20;
    const size_t count = static_cast<size_t>(n);
    const size_t bytes = count * sizeof(float);

    DeviceBuffer<float> d_a;
    DeviceBuffer<float> d_b;
    DeviceBuffer<float> d_c;
    if (!d_a.allocate(count, "performance cudaMalloc(d_a)") ||
        !d_b.allocate(count, "performance cudaMalloc(d_b)") ||
        !d_c.allocate(count, "performance cudaMalloc(d_c)")) {
        return false;
    }

    constexpr int fill_threads = 256;
    fill_buffer<<<(count + fill_threads - 1) / fill_threads, fill_threads>>>(
        d_a.get(), count, 1.0f / 16.0f);
    fill_buffer<<<(count + fill_threads - 1) / fill_threads, fill_threads>>>(
        d_b.get(), count, 1.0f / 16.0f);
    if (!cuda_ok(cudaGetLastError(), "performance input initialization") ||
        !cuda_ok(cudaDeviceSynchronize(),
                 "performance input initialization synchronize")) {
        return false;
    }

    for (int i = 0; i < warmup_iterations; ++i) {
        solve(d_a.get(), d_b.get(), d_c.get(), n);
    }
    if (!cuda_ok(cudaGetLastError(), "vector_add warmup") ||
        !cuda_ok(cudaDeviceSynchronize(), "vector_add warmup synchronize")) {
        return false;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "cudaEventCreate(start)") ||
        !cuda_ok(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        return false;
    }

    float total_ms = 0.0f;
    float minimum_ms = std::numeric_limits<float>::max();
    float maximum_ms = 0.0f;
    bool success = true;
    for (int i = 0; i < measured_iterations; ++i) {
        float elapsed_ms = 0.0f;
        if (!cuda_ok(cudaEventRecord(start), "record start")) {
            success = false;
            break;
        }
        solve(d_a.get(), d_b.get(), d_c.get(), n);
        if (!cuda_ok(cudaGetLastError(), "vector_add performance kernel") ||
            !cuda_ok(cudaEventRecord(stop), "record stop") ||
            !cuda_ok(cudaEventSynchronize(stop), "synchronize stop") ||
            !cuda_ok(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time")) {
            success = false;
            break;
        }
        total_ms += elapsed_ms;
        minimum_ms = std::min(minimum_ms, elapsed_ms);
        maximum_ms = std::max(maximum_ms, elapsed_ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!success) {
        return false;
    }

    const double average_ms = total_ms / measured_iterations;
    const double transferred_bytes = 3.0 * static_cast<double>(bytes);
    const double effective_bandwidth_gbs = transferred_bytes / (average_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(3)
              << "[PERF] vector_add N=" << n
              << ", iterations=" << measured_iterations
              << ", avg=" << average_ms << " ms"
              << ", max=" << maximum_ms << " ms"
              << ", min=" << minimum_ms << " ms"
              << ", effective bandwidth=" << effective_bandwidth_gbs << " GB/s\n";
    // ==================== Core performance test ends ====================
    return true;
}

int main() {
    constexpr std::array<int, 6> correctness_sizes = {
        1, 255, 256, 257, 4099, (1 << 20) + 37
    };

    bool success = true;
    std::cout << "=== Vector addition correctness ===\n";
    for (int n : correctness_sizes) {
        if (!run_correctness_case(n)) {
            success = false;
        }
    }

    std::cout << "=== Vector addition performance ===\n";
    if (!run_performance_test()) {
        success = false;
    }

    std::cout << (success ? "PASS: vector_add test suite\n"
                          : "FAIL: vector_add test suite\n");
    return success ? 0 : 1;
}
/*

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>

extern "C" void solve(const float* a, const float* b, float* c, int n);

int main() {
    constexpr int n = 4;
    const std::array<float, n> input_a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, n> input_b = {5.0f, 6.0f, 7.0f, 8.0f};
    const std::array<float, n> expected = {6.0f, 8.0f, 10.0f, 12.0f};
    std::array<float, n> actual = {};
    constexpr size_t bytes = n * sizeof(float);

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
                   check_cuda(cudaMemcpy(device_a, input_a.data(), bytes, cudaMemcpyHostToDevice), "copy input_a") &&
                   check_cuda(cudaMemcpy(device_b, input_b.data(), bytes, cudaMemcpyHostToDevice), "copy input_b");

    if (success) {
        solve(device_a, device_b, device_c, n);
        success = check_cuda(cudaGetLastError(), "solve") &&
                  check_cuda(cudaDeviceSynchronize(), "synchronize") &&
                  check_cuda(cudaMemcpy(actual.data(), device_c, bytes, cudaMemcpyDeviceToHost), "copy result");
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    if (!success) {
        return 1;
    }

    for (int index = 0; index < n; ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1e-6f) {
            std::cerr << "Test failed at index " << index
                      << ". Expected: " << expected[index]
                      << ", Actual: " << actual[index] << '\n';
            return 1;
        }
    }

    std::cout << "Test passed.\n";
    return 0;
}
*/
