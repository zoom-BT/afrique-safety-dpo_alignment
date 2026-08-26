import torch

from src.utils import get_device, set_seed


def test_set_seed_makes_random_reproducible():
    set_seed(42)
    first = torch.rand(3)
    set_seed(42)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_get_device_returns_a_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")
