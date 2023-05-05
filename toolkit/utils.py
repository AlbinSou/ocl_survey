#!/usr/bin/env python3
import random
from types import SimpleNamespace
from typing import Dict, Union

import numpy as np
import torch

from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheSubset


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False


def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    if additional_args is not None:
        for k, v in additional_args.items():
            args.__dict__[k] = v
    return args


def extract_kwargs(extract, kwargs):
    """
    checks and extracts
    the arguments
    listed in extract
    """
    init_dict = {}
    for word in extract:
        if word not in kwargs:
            raise AttributeError(f"Missing attribute {word} in provided configuration")
        init_dict.update({word: kwargs[word]})
    return init_dict

def assert_in(args, list):
    for arg in args:
        assert arg in list