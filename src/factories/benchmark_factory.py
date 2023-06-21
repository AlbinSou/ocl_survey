#!/usr/bin/env python3

import os
from typing import Any, Optional, Sequence, Union

from torchvision import transforms

from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.classic import (SplitCIFAR10, SplitCIFAR100,
                                          SplitImageNet, SplitTinyImageNet)
from src.factories.default_transforms import *


"""
Benchmarks factory
"""

DS_SIZES = {
    "split_imagenet": (256, 256, 3),
    "split_cifar100": (32, 32, 3),
    "split_tinyimagenet": (64, 64, 3),
}

DS_CLASSES = {
    "split_imagenet": 1000,
    "split_cifar100": 100,
    "split_tinyimagenet": 200,
}

def create_benchmark(
    benchmark_name: str,
    n_experiences: int,
    *,
    val_size: float = 0,
    seed: Optional[int] = None,
    dataset_root: Union[str] = None,
    first_exp_with_half_classes: bool = False,
    return_task_id=False,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    use_transforms: bool = True,
):
    benchmark = None
    if benchmark_name == "split_cifar100":
        if not use_transforms:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
                    ),
                ]
            )
            eval_transform = train_transform
        else:
            train_transform = default_cifar100_train_transform
            eval_transform = default_cifar100_eval_transform

        benchmark = SplitCIFAR100(
            n_experiences,
            first_exp_with_half_classes=first_exp_with_half_classes,
            return_task_id=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )
    elif benchmark_name == "split_cifar10":
        if not use_transforms:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            eval_transform = train_transform
        else:
            train_transform = default_cifar10_train_transform
            eval_transform = default_cifar10_eval_transform

        benchmark = SplitCIFAR10(
            n_experiences,
            first_exp_with_half_classes=first_exp_with_half_classes,
            return_task_id=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )

    elif benchmark_name == "split_imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if not use_transforms:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            eval_transform = train_transform
        else:
            train_transform = default_imagenet_train_transform
            eval_transform = default_imagenet_eval_transform

        benchmark = SplitImageNet(
            n_experiences=n_experiences,
            return_task_id=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )

    elif benchmark_name == "split_tinyimagenet":
        if not use_transforms:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            eval_transform = train_transform
        else:
            train_transform = default_tinyimagenet_train_transform
            eval_transform = default_cifar100_eval_transform

        benchmark = SplitTinyImageNet(
            n_experiences,
            return_task_id=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )

    assert benchmark is not None
    print(benchmark.classes_order_original_ids)

    if val_size > 0:
        benchmark = benchmark_with_validation_stream(
            benchmark, validation_size=val_size
        )

    return benchmark
