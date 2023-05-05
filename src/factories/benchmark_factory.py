#!/usr/bin/env python3

from typing import Optional, Sequence, Any, Union
from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100

"""
Benchmarks factory
"""

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
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
):
    benchmark = None
    if benchmark_name == "split_cifar100":
        benchmark = SplitCIFAR100(
            n_experiences,
            first_exp_with_half_classes=first_exp_with_half_classes,
            return_task_id=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )
    elif benchmark_name == "split_cifar10":
        benchmark = SplitCIFAR10(
            n_experiences,
            first_exp_with_half_classes=first_exp_with_half_classes,
            return_task_id=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
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
