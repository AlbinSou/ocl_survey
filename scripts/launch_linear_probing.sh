#!/bin/sh

python ../src/main.py --config-path $1 --config-name config.yaml experiment.save_models=false optimizer.type=SGD optimizer.lr=0.01 optimizer.weight_decay=0.0002 strategy.name=linear_probing strategy.train_epochs=100 strategy.train_mb_size=64 strategy.eval_mb_size=64 hydra.job.chdir=false hydra.output_subdir=null

