#!/bin/bash

# $1 strategy, $2 memory size

for SEED in 0 1 2 3 4;
do
    python ../experiments/main.py strategy="$1" +best_configs="$1" \
        strategy.mem_size=$2 experiment.seed=$SEED evaluation=parallel \
        benchmark=split_cifar100 \
        evaluation.num_gpus=0.2 evaluation.num_cpus=1 evaluation.num_actors=4
done
