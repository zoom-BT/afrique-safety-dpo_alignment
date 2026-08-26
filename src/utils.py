"""Environment setup, reproducibility, and lightweight GPU diagnostics."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_gpu_memory() -> str:
    if not torch.cuda.is_available():
        return "No GPU detected; running on CPU."
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return f"GPU memory allocated: {allocated:.1f} MB, reserved: {reserved:.1f} MB"
