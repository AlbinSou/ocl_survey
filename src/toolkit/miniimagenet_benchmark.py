#  Copyright (c) 2021-2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.datasets.mini_imagenet.mini_imagenet import \
    MiniImageNetDataset

_default_train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

_default_test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def SplitMiniImageNet(
    n_experiences=20,
    dataset_root=None,
    return_task_id=False,
    seed=0,
    fixed_class_order=None,
    shuffle=True,
    class_ids_from_zero_in_each_exp=False,
    class_ids_from_zero_from_first_exp=False,
    train_transform=_default_train_transform,
    eval_transform=_default_test_transform,
    preprocessed=False,
):
    """
    Creates a CL scenario using the Mini ImageNet dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param preprocessed: Use preprocessed images for Mini-Imagenet if True, otherwise use original Imagenet.
    :param dataset_root: Root path of the downloaded dataset.
    :param n_experiences: The number of experiences in the current scenario.
    :param return_task_id: if True, for every experience the task id is returned
        and the Scenario is Multi Task. This means that the scenario returned
        will be of type ``NCMultiTaskScenario``. If false the task index is
        not returned (default to 0 for every batch) and the returned scenario
        is of type ``NCSingleTaskScenario``.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param test_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the the
        MT scenario if the parameter ``return_task_id`` is True,
        a :class:`NCSingleTaskScenario` initialized for the SIT scenario otherwise.
    """

    if preprocessed:
        print("not available")
    else:
        train_set, test_set = _get_mini_imagenet_dataset(os.path.join(dataset_root))

    return nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        shuffle=shuffle,
        fixed_class_order=fixed_class_order,
        per_exp_classes=None,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def _get_mini_imagenet_dataset(path):
    """Create from ImageNet."""
    train_set = MiniImageNetDataset(path, split="all")

    train_set_images = np.array([np.array(img[0]) for img in train_set])
    train_set_labels = np.array(train_set.targets)

    train_x, test_x = [], []
    train_y, test_y = [], []

    for target in np.unique(train_set.targets):
        subset_x = train_set_images[train_set_labels == target]
        subset_y = train_set_labels[train_set_labels == target]
        train_x.extend(subset_x[:500])
        test_x.extend(subset_x[500:])
        train_y.extend(subset_y[:500])
        test_y.extend(subset_y[500:])

    return XYDataset(train_x, train_y), XYDataset(test_x, test_y)


class XYDataset(Dataset):
    """Template Dataset with Labels"""

    def __init__(self, x, y, **kwargs):
        self.x, self.targets = x, y
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.targets[idx]
        return x, y


__all__ = ["SplitMiniImageNet"]
