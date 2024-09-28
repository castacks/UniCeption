import torch.utils.benchmark as benchmark


def benchmark_torch_function(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    return t0.blocked_autorange().mean * 1e3  # Milliseconds


def benchmark_torch_function_with_result(f, *args, **kwargs):
    result = f(*args, **kwargs)
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    time_in_ms = t0.blocked_autorange().mean * 1e3  # Milliseconds
    return time_in_ms, result
