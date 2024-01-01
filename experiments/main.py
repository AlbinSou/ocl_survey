#!/usr/bin/env python3
import argparse
import os

import hydra
import numpy as np
import omegaconf
import ray
import torch

import avalanche.benchmarks.scenarios as scenarios
import src.factories.benchmark_factory as benchmark_factory
import src.factories.method_factory as method_factory
import src.factories.model_factory as model_factory
import src.toolkit.utils as utils
from avalanche.benchmarks.scenarios import OnlineCLScenario
from src.factories.benchmark_factory import DS_SIZES


@hydra.main(config_path="../config", config_name="config.yaml")
def main(config):
    utils.set_seed(config.experiment.seed)

    plugins = []

    scenario = benchmark_factory.create_benchmark(
        **config["benchmark"].factory_args,
        dataset_root=config.benchmark.dataset_root,
    )

    model = model_factory.create_model(
        **config["model"],
        input_size=DS_SIZES[config.benchmark.factory_args.benchmark_name],
    )

    optimizer, scheduler_plugin = model_factory.get_optimizer(
        model,
        optimizer_type=config.optimizer.type,
        scheduler_type=config.scheduler.type,
        kwargs_optimizer=config["optimizer"],
        kwargs_scheduler=config["scheduler"],
    )
    print(optimizer)

    if scheduler_plugin is not None:
        plugins.append(scheduler_plugin)

    exp_name = (
        config.strategy.name
        + "_"
        + config.benchmark.factory_args.benchmark_name
        + "_"
        + str(config.benchmark.factory_args.n_experiences)
        + "_"
        + str(config.strategy.mem_size)
    )

    if not config.experiment.debug:
        logdir = os.path.join(
            str(config.experiment.results_root),
            exp_name,
            str(config.experiment.seed),
        )
    else:
        logdir = os.path.join(
            str(config.experiment.results_root),
            "debug",
        )

    if config.experiment.logdir is None:
        os.makedirs(logdir, exist_ok=True)
        utils.clear_tensorboard_files(logdir)

        # Add full results dir to config
        config.experiment.logdir = logdir

        omegaconf.OmegaConf.save(config, os.path.join(logdir, "config.yaml"))
    else:
        logdir = config.experiment.logdir

    strategy = method_factory.create_strategy(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        logdir=logdir,
        name=config.strategy.name,
        dataset_name=config.benchmark.factory_args.benchmark_name,
        strategy_kwargs=config["strategy"],
        evaluation_kwargs=config["evaluation"],
    )

    print("Using strategy: ", strategy.__class__.__name__)
    print("With plugins: ", strategy.plugins)

    batch_streams = scenario.streams.values()
    for t, experience in enumerate(scenario.train_stream):
        if config.experiment.train_online:
            ocl_scenario = OnlineCLScenario(
                original_streams=batch_streams,
                experiences=experience,
                experience_size=config.strategy.train_mb_size,
                access_task_boundaries=config.strategy.use_task_boundaries,
            )
            train_stream = ocl_scenario.train_stream
        else:
            train_stream = experience

        strategy.train(
            train_stream,
            eval_streams=[scenario.valid_stream[: t + 1]],
            num_workers=0,
            drop_last=True,
            reset_optimizer_state=False,
        )

        if config.experiment.save_models:
            torch.save(
                strategy.model.state_dict(), os.path.join(logdir, f"model_{t}.ckpt")
            )

        results = strategy.eval(scenario.test_stream[: t + 1])

    return results


if __name__ == "__main__":
    main()
