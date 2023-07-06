#!/bin/sh

python ../experiments/main.py --config-path $1 --config-name config.yaml \
    strategy.name=linear_probing \
    strategy.train_epochs=500 \
    strategy.train_mb_size=128 \
    optimizer.type=AdamW \
    optimizer.lr=0.001 \
    optimizer.weight_decay=0.0002 \
    hydra.output_subdir=null \
    hydra.job.chdir=false \
    experiment.train_online=false \
    experiment.save_models=false \
    evaluation.parallel_evaluation=false \
    evaluation.eval_every=-1 \
