"""
This file compares the speed of transformer engine v.s. UniCeption Attention Implementation.
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

from uniception.models.utils.transformer_blocks import SelfAttentionBlock


def cuda_profile(profile_func, warmup: int = 10, test: int = 100):
    """
    Run a CUDA‑backed callable several times and report timing statistics.

    Parameters
    ----------
    profile_func : Callable[[], Any]
        A function that launches the CUDA work you want to measure.
        It should *not* include its own synchronizations.
    warmup : int
        Number of warm‑up iterations to ignore in the timing results.
    test : int
        Number of timed iterations.

    Returns
    -------
    dict
        {
            'avg_ms':   mean latency per iteration (milliseconds),
            'std_ms':   un‑biased std‑dev of latency (milliseconds),
            'total_ms': total measured wall‑clock time (milliseconds),
            'iters_per_sec': throughput in iterations / second
        }
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    # Ensure the first kernel launch won’t be timed
    torch.cuda.synchronize()

    # -------------------
    # Warm‑up phase
    # -------------------
    for _ in range(warmup):
        profile_func()
    torch.cuda.synchronize()

    # -------------------
    # Timed phase
    # -------------------
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times_ms = torch.empty(test, dtype=torch.float64, device="cpu")

    for i in range(test):
        start_evt.record()
        profile_func()
        end_evt.record()
        torch.cuda.synchronize()  # wait for kernels
        times_ms[i] = start_evt.elapsed_time(end_evt)  # ms for this iter

    # -------------------
    # Aggregate stats
    # -------------------
    avg = times_ms.mean().item()
    std = times_ms.std(unbiased=True).item()
    total = times_ms.sum().item()

    return {
        "avg_ms": avg,
        "std_ms": std,
        "total_ms": total,
        "iters_per_sec": 1_000.0 / avg if avg > 0 else float("inf"),
    }


def te_fused(batch_size: int, head_dim: int, n_head: int, seq_len: int, precision: str = "bfloat16"):
    """
    Profile the Transformer Engine fused kernel
    """

    te_transformer = (
        te.TransformerLayer(
            head_dim * n_head,
            4 * head_dim * n_head,
            n_head,
            fuse_qkv_params=True,
            self_attn_mask_type="no_mask",
            hidden_dropout=0,
            attention_dropout=0,
            window_size=(-1, -1),
        )
        .cuda()
        .to(dtype=torch.bfloat16)
    )

    seq_len_list = [
        seq_len,
        int(1.25 * seq_len),
        int(1.5 * seq_len),
        int(1.75 * seq_len),
        int(2.0 * seq_len),
        int(0.75 * seq_len),
        int(0.5 * seq_len),
    ]
    x_list = [torch.rand(s, batch_size, head_dim * n_head).cuda().to(dtype=torch.bfloat16) for s in seq_len_list]

    if precision == "bfloat16":

        def profile_func():
            for x in x_list:
                y = te_transformer(x)
                torch.sum(y).backward()

    elif precision == "fp8":
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

        def profile_func():
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                for x in x_list:
                    y = te_transformer(x)
                    torch.sum(y).backward()

    result = cuda_profile(profile_func, 5, 100)

    return result


def uniception(batch_size: int, head_dim: int, n_head: int, seq_len: int, precision: str = "bfloat16"):

    attention = torch.compile(
        SelfAttentionBlock(dim=head_dim * n_head, num_heads=n_head, qkv_bias=True).cuda().to(dtype=torch.bfloat16),
        dynamic=True,
        fullgraph=True,
    )

    seq_len_list = [
        seq_len,
        int(1.25 * seq_len),
        int(1.5 * seq_len),
        int(1.75 * seq_len),
        int(2.0 * seq_len),
        int(0.75 * seq_len),
        int(0.5 * seq_len),
    ]
    x_list = [torch.rand(s, batch_size, head_dim * n_head).cuda().to(dtype=torch.bfloat16) for s in seq_len_list]

    def profile_func():
        for x in x_list:
            y = attention(x)
            torch.sum(y).backward()

    result = cuda_profile(profile_func, 5, 100)

    return result


if __name__ == "__main__":

    import pandas as pd

    dataframe_data = []

    # batch_sizes = [16, 32, 64, 128, 256]
    # seq_lengths = [512, 1024, 2048, 3072]

    batch_sizes = [16]
    seq_lengths = [512]

    for bs in batch_sizes:
        for seq_len in seq_lengths:
            # test te bfloat16
            results_te_fused = te_fused(bs, 128, 8, seq_len)
            results_te_fused["method"] = "te"
            results_te_fused["precision"] = "bfloat16"
            results_te_fused["batch_size"] = bs
            results_te_fused["seq_length"] = seq_len

            dataframe_data.append(results_te_fused)

            # test UniCeption bfloat16
            results_uc = uniception(bs, 128, 8, seq_len)
            results_uc["method"] = "uniception"
            results_uc["precision"] = "bfloat16"
            results_uc["batch_size"] = bs
            results_uc["seq_length"] = seq_len

            dataframe_data.append(results_uc)

            # test te fp8
            results_te_fused_fp8 = te_fused(bs, 128, 8, seq_len, precision="fp8")
            results_te_fused_fp8["method"] = "te"
            results_te_fused_fp8["precision"] = "fp8"
            results_te_fused_fp8["batch_size"] = bs
            results_te_fused_fp8["seq_length"] = seq_len

            dataframe_data.append(results_te_fused_fp8)

    dataframe = pd.DataFrame(dataframe_data)
    dataframe.to_csv("dataframe.csv", index=False)
