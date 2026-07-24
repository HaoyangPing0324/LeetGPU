import importlib.util
import sys
from pathlib import Path

import torch


def load_implementation():
    source_path = Path(__file__).resolve().parents[2] / "src" / "pytorch" / "01_Vector_Addition.py"
    spec = importlib.util.spec_from_file_location("vector_addition_pytorch", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_correctness_tests(solve):
    sizes = [1, 255, 256, 257, 4099, (1 << 20) + 37]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2026)

    for size in sizes:
        input_a = torch.randn(size, device="cuda", dtype=torch.float32, generator=generator)
        input_b = torch.randn(size, device="cuda", dtype=torch.float32, generator=generator)
        actual = torch.empty_like(input_a)
        expected = torch.add(input_a, input_b)

        solve(input_a, input_b, actual, size)
        torch.cuda.synchronize()
        if not torch.isfinite(actual).all():
            raise AssertionError("Output contains non-finite values.")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        print(f"[PASS] vector_add correctness N={size}")


def run_performance_test(solve):
    size = 25_000_000
    warmup_iterations = 5
    measured_iterations = 20
    pattern = (torch.arange(17, device="cuda", dtype=torch.float32) - 8.0) / 16.0
    input_a = pattern.repeat((size + 16) // 17)[:size]
    input_b = pattern.repeat((size + 16) // 17)[:size]
    output = torch.empty_like(input_a)

    for _ in range(warmup_iterations):
        solve(input_a, input_b, output, size)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(measured_iterations):
        start.record()
        solve(input_a, input_b, output, size)
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop))

    average = sum(timings) / len(timings)
    maximum = max(timings)
    minimum = min(timings)
    transferred_bytes = 3 * size * torch.tensor([], dtype=torch.float32).element_size()
    bandwidth = transferred_bytes / (average * 1.0e6)
    print(
        f"[PERF] vector_add N={size}, iterations={measured_iterations}, "
        f"avg={average:.3f} ms, max={maximum:.3f} ms, min={minimum:.3f} ms, "
        f"effective bandwidth={bandwidth:.3f} GB/s"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    implementation = load_implementation()
    print("=== Vector addition correctness ===")
    run_correctness_tests(implementation.solve)
    print("=== Vector addition performance ===")
    run_performance_test(implementation.solve)
    print("PASS: vector_add test suite")


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
    source_path = Path(__file__).resolve().parents[2] / "src" / "pytorch" / "01_Vector_Addition.py"
    spec = importlib.util.spec_from_file_location("vector_addition_pytorch_minimal", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    input_a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    input_b = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda")
    expected = torch.tensor([6.0, 8.0, 10.0, 12.0], device="cuda")
    actual = torch.empty_like(input_a)

    load_implementation().solve(input_a, input_b, actual, input_a.numel())
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
