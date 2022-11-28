from datetime import datetime
import os

import numpy as np
import torch


def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_timestamp():
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


def create_dir(path_prefix, dir_name):
    output_dir = os.path.join(path_prefix, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
