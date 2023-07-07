#!/bin/sh

# Same compute time with bs 128 = 38 epochs

python ../experiments/main.py experiment=split_cifar100 strategy=gdumb experiment.train_online=false strategy.train_mb_size=128 strategy.train_epochs=100 experiment.debug=false scheduler=cosine scheduler.T_max=100 scheduler.step_granularity=epoch
