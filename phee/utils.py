import contextlib
import gc
import os
import random
import time

import numpy as np


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(precision):
    """bf16 autocast on GPUs that support it (RTX 30xx/40xx/50xx); fp32 otherwise."""
    import torch

    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class GpuMemory:
    """Peak GPU memory (GB) of a code block; None on CPU."""

    def __init__(self):
        self.peak_gb = None

    def __enter__(self):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        return False


class Timer:
    def __init__(self):
        self.seconds = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._start
        return False


def cleanup():
    """Release cached GPU memory (call after `del model`)."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
