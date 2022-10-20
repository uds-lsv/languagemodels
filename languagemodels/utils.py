import torch


def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
