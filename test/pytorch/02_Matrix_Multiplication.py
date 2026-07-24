import importlib.util
import sys
from pathlib import Path

import torch


def load_implementation():
    source_path = Path(__file__).resolve().parents[2] / "src" / "pytorch" / "02_Matrix_Multiplication.py"
    spec = importlib.util.spec_from_file_location("matrix_multiplication_pytorch", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_correctness_tests(solve):
    shapes = [
        (1, 1, 1),
        (2, 2, 2),
        (1, 3, 1),
        (17, 19, 23),
        (127, 129, 131),
        (64, 96, 128),
        (128, 128, 128),
    ]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2026)

    for m, n, k in shapes:
        matrix_a = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=generator)
        matrix_b = torch.randn((n, k), device="cuda", dtype=torch.float32, generator=generator)
        actual = torch.empty((m, k), device="cuda", dtype=torch.float32)
        expected = torch.matmul(matrix_a, matrix_b)

        solve(matrix_a, matrix_b, actual, m, n, k)
        torch.cuda.synchronize()
        if not torch.isfinite(actual).all():
            raise AssertionError("Output contains non-finite values.")
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        print(f"[PASS] matrix correctness {m}x{n} * {n}x{k}")


def run_performance_test(solve):
    m, n, k = 8192, 6144, 4096
    warmup_iterations = 5
    measured_iterations = 10
    pattern = (torch.arange(17, device="cuda", dtype=torch.float32) - 8.0) / 16.0
    matrix_a = pattern.repeat((m * n + 16) // 17)[:m * n].reshape(m, n)
    matrix_b = pattern.repeat((n * k + 16) // 17)[:n * k].reshape(n, k)
    output = torch.empty((m, k), device="cuda", dtype=torch.float32)

    for _ in range(warmup_iterations):
        solve(matrix_a, matrix_b, output, m, n, k)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(measured_iterations):
        start.record()
        solve(matrix_a, matrix_b, output, m, n, k)
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop))

    average = sum(timings) / len(timings)
    maximum = max(timings)
    minimum = min(timings)
    operations = 2.0 * m * n * k
    throughput = operations / (average * 1.0e6)
    print(
        f"[PERF] matrix M={m}, N={n}, K={k}, iterations={measured_iterations}, "
        f"avg={average:.3f} ms, max={maximum:.3f} ms, min={minimum:.3f} ms, "
        f"throughput={throughput:.3f} GFLOPS"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    implementation = load_implementation()
    print("=== Matrix multiplication correctness ===")
    run_correctness_tests(implementation.solve)
    print("=== Matrix multiplication performance ===")
    run_performance_test(implementation.solve)
    print("PASS: matrix multiplication test suite")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise
'''

import importlib.util
import sys
from pathlib import Path

import torch


def load_implementation():
    source_path = Path(__file__).resolve().parents[2] / "src" / "pytorch" / "02_Matrix_Multiplication.py"
    spec = importlib.util.spec_from_file_location("matrix_multiplication_pytorch_minimal", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    matrix_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    matrix_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device="cuda")
    actual = torch.empty_like(expected)

    load_implementation().solve(matrix_a, matrix_b, actual, 2, 2, 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)
    print("Test passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Test failed: {error}", file=sys.stderr)
        raise
'''
