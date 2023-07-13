#!/bin/sh

# Basepath $1

for SEED in 0 1 2 3 4;
do

    python ../experiments/main.py --config-path "$1/$SEED" --config-name config.yaml \
        strategy.name=linear_probing \
        strategy.train_mb_size=128 \
        hydra.output_subdir=null \
        hydra.job.chdir=false \
        experiment.train_online=false \
        experiment.save_models=false \
        evaluation.parallel_evaluation=false \
        evaluation.eval_every=-1

done
