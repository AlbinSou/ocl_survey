#!/usr/bin/env python3

import time

from avalanche.training.storage_policy import ClassBalancedBuffer
from src.factories.benchmark_factory import create_benchmark


def test_imnet_balanced_buff():
    scenario = create_benchmark(
        "split_imagenet",
        20,
        dataset_root="/DATA/data/",
        shuffle=True,
        use_transforms=True,
    )

    storage_policy = ClassBalancedBuffer(10000)

    for experience in scenario.train_stream:
        a = time.time()
        storage_policy.update_from_dataset(experience.dataset)
        b = time.time()
        print("Update from dataset:", b - a)

        a = time.time()
        storage_policy.buffer
        b = time.time()
        print("Access buffer", b - a)


if __name__ == "__main__":
    test_imnet_balanced_buff()
