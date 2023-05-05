#!/usr/bin/env python3
import os

import hydra
import omegaconf
import ray

import avalanche.benchmarks.scenarios as scenarios
import src.factories.benchmark_factory as benchmark_factory
import src.factories.method_factory as method_factory
import src.factories.model_factory as model_factory
import toolkit.utils as utils


@hydra.main(config_path="../config")
def main(config):
    utils.set_seed(config.experiment.seed)

    plugins = []

    scenario = benchmark_factory.create_benchmark(**config["benchmark"])

    model = model_factory.create_model(**config["model"])

    optimizer, scheduler_plugin = model_factory.get_optimizer(
        model,
        optimizer_type=config.optimizer.type,
        scheduler_type=config.scheduler.type,
        kwargs_optimizer=config["optimizer"],
        kwargs_scheduler=config["scheduler"],
    )
    if scheduler_plugin is not None:
        plugins.append(scheduler_plugin)

    exp_name = config.strategy.name + "_" + config.benchmark.benchmark_name

    logdir = os.path.join(
        str(config.experiment.results_root),
        exp_name,
        str(config.experiment.seed),
    )

    if not os.path.isdir(os.path.join(str(config.experiment.results_root), exp_name)):
        os.mkdir(os.path.join(str(config.experiment.results_root), exp_name))
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    omegaconf.OmegaConf.save(config, os.path.join(logdir, "config.yml"))

    strategy = method_factory.create_strategy(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        logdir=logdir,
        name=config.strategy.name,
        strategy_kwargs=config["strategy"],
        evaluation_kwargs=config["evaluation"],
    )

    print("Using strategy: ", strategy.__class__.__name__)
    print("With plugins: ", plugins)


if __name__ == "__main__":
    main()
