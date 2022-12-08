import argparse
from datetime import datetime
import json
import os
from typing import Union

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


def save_args(args: argparse.Namespace, path: Union[str, os.PathLike]) -> None:
    out_file = os.path.join(path, "args.json")
    with open(out_file, 'w') as f:
        args_dict = args.__dict__
        out_str = json.dumps(args_dict)
        json.dump(out_str, f)


def load_args(path: Union[str, os.PathLike]) -> argparse.Namespace:
    in_file = os.path.join(path, "args.json")
    with open(in_file, 'r') as f:
        args_str = json.load(f)
        args_dict = json.loads(args_str)
    return argparse.Namespace(**args_dict)
