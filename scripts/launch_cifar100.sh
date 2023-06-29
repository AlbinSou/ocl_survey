#!/bin/bash

# $1 strategy, $2 memory size

for SEED in 0 1 2 3 4;
do
    python ../experiments/main.py strategy="$1" +best_configs=split_cifar100/$1 \
        strategy.mem_size=$2 experiment.seed=$SEED experiment.save_models=true \
        evaluation=parallel experiment=split_cifar100 deploy=default
done
