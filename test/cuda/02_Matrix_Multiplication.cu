#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

extern "C" void solve(const float* a, const float* b, float* c,
                      int m, int n, int k);

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

struct MatrixShape {
    int m;
    int n;
    int k;
};

static bool run_correctness_case(const MatrixShape& shape) {
    const int m = shape.m;
    const int n = shape.n;
    const int k = shape.k;
    const size_t a_count = static_cast<size_t>(m) * n;
    const size_t b_count = static_cast<size_t>(n) * k;
    const size_t c_count = static_cast<size_t>(m) * k;
    std::vector<float> a(a_count);
    std::vector<float> b(b_count);
    std::vector<float> c(c_count);

    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(static_cast<int>((i * 17) % 31) - 15) / 17.0f;
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(static_cast<int>((i * 13) % 29) - 14) / 13.0f;
    }

    DeviceBuffer<float> d_a;
    DeviceBuffer<float> d_b;
    DeviceBuffer<float> d_c;
    if (!d_a.allocate(a_count, "cudaMalloc(d_a)") ||
        !d_b.allocate(b_count, "cudaMalloc(d_b)") ||
        !d_c.allocate(c_count, "cudaMalloc(d_c)") ||
        !cuda_ok(cudaMemcpy(d_a.get(), a.data(), a_count * sizeof(float),
                            cudaMemcpyHostToDevice), "copy a") ||
        !cuda_ok(cudaMemcpy(d_b.get(), b.data(), b_count * sizeof(float),
                            cudaMemcpyHostToDevice), "copy b")) {
        return false;
    }

    solve(d_a.get(), d_b.get(), d_c.get(), m, n, k);
    if (!cuda_ok(cudaGetLastError(), "matrix multiplication kernel") ||
        !cuda_ok(cudaDeviceSynchronize(), "matrix multiplication synchronize") ||
        !cuda_ok(cudaMemcpy(c.data(), d_c.get(), c_count * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy c")) {
        return false;
    }

    int errors = 0;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            float expected = 0.0f;
            for (int inner = 0; inner < n; ++inner) {
                expected += a[static_cast<size_t>(row) * n + inner] *
                            b[static_cast<size_t>(inner) * k + col];
            }
            const float actual = c[static_cast<size_t>(row) * k + col];
            const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected));
            if (std::fabs(actual - expected) > tolerance) {
                if (errors < 5) {
                    std::cerr << "Mismatch at (" << row << ", " << col << "): "
                              << actual << " != " << expected << '\n';
                }
                ++errors;
            }
        }
    }

    if (errors != 0) {
        std::cerr << "[FAIL] matrix correctness " << m << 'x' << n
                  << " * " << n << 'x' << k
                  << ", mismatches=" << errors << '\n';
        return false;
    }

    std::cout << "[PASS] matrix correctness " << m << 'x' << n
              << " * " << n << 'x' << k << '\n';
    return true;
}

static bool run_performance_test() {
    constexpr int m = 8192;
    constexpr int n = 6144;
    constexpr int k = 4096;
    constexpr int warmup_iterations = 1;
    constexpr int measured_iterations = 3;
    const size_t a_count = static_cast<size_t>(m) * n;
    const size_t b_count = static_cast<size_t>(n) * k;
    const size_t c_count = static_cast<size_t>(m) * k;

    DeviceBuffer<float> d_a;
    DeviceBuffer<float> d_b;
    DeviceBuffer<float> d_c;
    if (!d_a.allocate(a_count, "performance cudaMalloc(d_a)") ||
        !d_b.allocate(b_count, "performance cudaMalloc(d_b)") ||
        !d_c.allocate(c_count, "performance cudaMalloc(d_c)") ||
        !cuda_ok(cudaMemset(d_a.get(), 0, a_count * sizeof(float)), "performance memset a") ||
        !cuda_ok(cudaMemset(d_b.get(), 0, b_count * sizeof(float)), "performance memset b")) {
        return false;
    }

    for (int i = 0; i < warmup_iterations; ++i) {
        solve(d_a.get(), d_b.get(), d_c.get(), m, n, k);
    }
    if (!cuda_ok(cudaGetLastError(), "matrix warmup") ||
        !cuda_ok(cudaDeviceSynchronize(), "matrix warmup synchronize")) {
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
        solve(d_a.get(), d_b.get(), d_c.get(), m, n, k);
        if (!cuda_ok(cudaGetLastError(), "matrix performance kernel") ||
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
    const double operations = 2.0 * static_cast<double>(m) * n * k;
    const double gflops = operations / (average_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(3)
              << "[PERF] matrix M=" << m << ", N=" << n << ", K=" << k
              << ", iterations=" << measured_iterations
              << ", avg=" << average_ms << " ms"
              << ", max=" << maximum_ms << " ms"
              << ", min=" << minimum_ms << " ms"
              << ", throughput=" << gflops << " GFLOPS\n";
    return true;
}

int main() {
    constexpr std::array<MatrixShape, 5> correctness_shapes = {{
        {1, 1, 1},
        {2, 2, 2},
        {1, 3, 1},
        {17, 19, 23},
        {127, 129, 131}
    }};

    bool success = true;
    std::cout << "=== Matrix multiplication correctness ===\n";
    for (const MatrixShape& shape : correctness_shapes) {
        if (!run_correctness_case(shape)) {
            success = false;
        }
    }

    std::cout << "=== Matrix multiplication performance ===\n";
    if (!run_performance_test()) {
        success = false;
    }

    std::cout << (success ? "PASS: matrix multiplication test suite\n"
                          : "FAIL: matrix multiplication test suite\n");
    return success ? 0 : 1;
}
