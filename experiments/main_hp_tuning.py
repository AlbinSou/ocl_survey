#!/usr/bin/env python3
import os

import hydra
import numpy as np
import omegaconf
import ray
from ray import tune
from ray.air import session
from ray.tune.search.hyperopt import HyperOptSearch

import avalanche.benchmarks.scenarios as scenarios
import src.factories.benchmark_factory as benchmark_factory
import src.factories.method_factory as method_factory
import src.factories.model_factory as model_factory
import src.toolkit.utils as utils
from avalanche.benchmarks.scenarios import OnlineCLScenario


def update_config(ray_config, config):
    for key, item in ray_config.items():
        config[key].update(item)


@hydra.main(config_path="../config", config_name="hp_config.yaml")
def main(config):
    space = {
        "optimizer": {
            "lr": tune.loguniform(1e-3, 1.0),
        },
        "strategy": {
            "train_epochs": tune.randint(1, 10),
            "lr_ramp": tune.sample_from(
                lambda spec: float(np.exp(
                    np.random.uniform(
                        np.log(1e-8), np.log((spec.config.optimizer.lr - 1e-5) / 1500)
                    )
                ))
            ),
        },
    }

    ray.init(num_gpus=1, num_cpus=12)

    hyperopt_search = HyperOptSearch(metric="final_accuracy", mode="max")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_function, config=config),
            {"gpu": 0.15, "num_retries": 0},
        ),
        tune_config=tune.TuneConfig(
            num_samples=50, max_concurrent_trials=8, search_alg=hyperopt_search
        ),
        param_space=space,
    )

    results = tuner.fit()


def train_function(ray_config, config):
    # Update config with ray args
    update_config(ray_config, config)

    if "T_max" in config["scheduler"]:
        config.scheduler.T_max = config.strategy.train_epochs + 1
    if "lr_ramp" in config["strategy"]:
        # Turn log uniform ramp (positive) to negative
        config.strategy.lr_ramp *= -1

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

    # if not os.path.isdir(os.path.join(str(config.experiment.results_root), exp_name)):
    #    os.mkdir(os.path.join(str(config.experiment.results_root), exp_name))
    # if not os.path.isdir(logdir):
    #    os.mkdir(logdir)

    # omegaconf.OmegaConf.save(config, os.path.join(logdir, "config.yml"))

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
    print("With plugins: ", strategy.plugins)

    batch_streams = scenario.streams.values()
    for t, experience in enumerate(
        scenario.train_stream[: config.experiment.stop_at_experience + 1]
    ):
        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=config.strategy.train_mb_size,
            access_task_boundaries=False,
        )

        strategy.train(
            ocl_scenario.train_stream,
            eval_streams=[],
            num_workers=0,
            drop_last=True,
        )

        results = strategy.eval(scenario.valid_stream[: t + 1])

    session.report(
        {"final_accuracy": results["Top1_Acc_Stream/eval_phase/valid_stream/Task000"]}
    )


if __name__ == "__main__":
    main()