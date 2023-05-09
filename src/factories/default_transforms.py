#!/usr/bin/env python3
from torchvision import transforms 

default_cifar100_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ]
)

default_cifar100_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ]
)

default_cifar10_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

default_cifar10_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)
