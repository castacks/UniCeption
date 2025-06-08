"""
Utility functions for UniCeption Encoders.
"""

import functools

import numpy as np
import torch


def profile_encoder(num_warmup=3, num_runs=20, autocast_precision="float16", use_compile=False, dynamic=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            device = "cuda"
            autocast_dtype = getattr(torch, autocast_precision)

            # Compile the model if requested
            if use_compile:
                compiled_func = torch.compile(func, dynamic=dynamic, mode="max-autotune")
            else:
                compiled_func = func

            with torch.autocast("cuda", dtype=autocast_dtype):
                # Warm-up runs
                for _ in range(num_warmup):
                    output = compiled_func(self, *args, **kwargs)
                    if isinstance(output, torch.Tensor):
                        output.sum().backward()
                    else:
                        output.features.sum().backward()
                    torch.cuda.synchronize()

                # Clear memory cache
                torch.cuda.empty_cache()

                # Lists to store results
                forward_times, backward_times, memory_usages = [], [], []

                for _ in range(num_runs):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    torch.cuda.reset_peak_memory_stats()
                    memory_before = torch.cuda.max_memory_allocated(device)

                    # Forward pass
                    start_event.record()
                    output = compiled_func(self, *args, **kwargs)
                    end_event.record()
                    torch.cuda.synchronize()
                    forward_times.append(start_event.elapsed_time(end_event))

                    # Backward pass
                    start_event.record()
                    if isinstance(output, torch.Tensor):
                        output.sum().backward()
                    else:
                        output.features.sum().backward()
                    end_event.record()
                    torch.cuda.synchronize()
                    backward_times.append(start_event.elapsed_time(end_event))

                    memory_after = torch.cuda.max_memory_allocated(device)
                    memory_usages.append((memory_after - memory_before) / 1e6)  # Convert to MB

            # Compute mean and standard deviation
            fwd_mean, fwd_std = np.mean(forward_times), np.std(forward_times)
            bwd_mean, bwd_std = np.mean(backward_times), np.std(backward_times)
            mem_mean, mem_std = np.mean(memory_usages), np.std(memory_usages)

            compile_status = (
                "with torch.compile (dynamic=True)"
                if use_compile and dynamic
                else "with torch.compile (dynamic=False)" if use_compile else "without torch.compile"
            )
            print(f"Profiling results {compile_status}:")
            print(f"Forward Pass Time: {fwd_mean:.2f} ± {fwd_std:.2f} ms")
            print(f"Backward Pass Time: {bwd_mean:.2f} ± {bwd_std:.2f} ms")
            print(f"Peak GPU Memory Usage: {mem_mean:.2f} ± {mem_std:.2f} MB")

            return output

        return wrapper

    return decorator
