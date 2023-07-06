#!/usr/bin/env python3
import random
from types import SimpleNamespace
from typing import Dict, Union

import numpy as np
import torch
import os

from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.models.dynamic_modules import IncrementalClassifier


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


def get_last_layer_name(model):
    last_layer_name = list(model.named_parameters())[-1][0].split(".")[0]

    last_layer = getattr(model, last_layer_name)

    if isinstance(last_layer, IncrementalClassifier):
        in_features = last_layer.classifier.in_features
    else:
        in_features = last_layer.in_features
    return last_layer_name, in_features


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

def map_args(kwargs, keys):
    """
    Maps keys1 to keys2 in kwargs
    """
    for k1, k2 in keys.items():
        assert k1 in kwargs
        value = kwargs.pop(k1)
        kwargs[k2] = value

def clear_tensorboard_files(directory):
    for root, name, files in os.walk(directory):
        for f in files:
            if "events" in f:
                os.remove(os.path.join(root, f))

def assert_in(args, list):
    for arg in args:
        assert arg in list
