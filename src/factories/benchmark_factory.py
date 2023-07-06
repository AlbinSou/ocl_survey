#!/usr/bin/env python3

import os
from typing import Any, Optional, Sequence, Union

from torchvision import transforms

from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.classic import (SplitCIFAR10, SplitCIFAR100,
                                          SplitImageNet, SplitTinyImageNet)
                                          
from src.factories.default_transforms import *
from src.toolkit.miniimagenet_benchmark import SplitMiniImageNet


"""
Benchmarks factory
"""

DS_SIZES = {
    "split_imagenet": (224, 224, 3),
    "split_cifar100": (32, 32, 3),
    "split_tinyimagenet": (64, 64, 3),
    "split_miniimagenet": (84, 84, 3),
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
            train_transform = default_cifar100_eval_transform
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
            train_transform = default_cifar10_eval_transform
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
            train_transform = default_imagenet_eval_transform
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
            eval_transform = default_tinyimagenet_eval_transform

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

    elif benchmark_name == "split_miniimagenet":
        if not use_transforms:
            train_transform = default_miniimagenet_eval_transform 
            eval_transform = train_transform
        else:
            train_transform = default_miniimagenet_train_transform
            eval_transform = default_miniimagenet_eval_transform

        benchmark = SplitMiniImageNet(
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
            benchmark, validation_size=val_size, shuffle=True
        )

    return benchmark
