#!/bin/sh

python ../experiments/main.py --config-path $1 --config-name config.yaml \
    experiment.save_models=false optimizer.type=AdamW optimizer.lr=0.001 \
    optimizer.weight_decay=0.0002 strategy.name=linear_probing \
    strategy.train_epochs=50 strategy.train_mb_size=64 strategy.eval_mb_size=64 \
    experiment.train_online=false hydra.job.chdir=false hydra.output_subdir=null
