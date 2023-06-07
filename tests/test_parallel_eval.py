#!/usr/bin/env python3
import atexit
import copy
import os
import time
from typing import Dict, List, Tuple

import ray
import torch
from torch.optim import SGD

from avalanche.benchmarks import SplitMNIST, benchmark_with_validation_stream
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from toolkit.json_logger import JSONLogger
from toolkit.parallel_eval import ParallelEvaluationPlugin
from avalanche.benchmarks.scenarios import OnlineCLScenario


def create_strategy(model, plugins, use_logger=False, **kwargs):
    optimizer = SGD(model.parameters(), lr=0.01)

    # NOTE: eval_every must be -1 to disable the main strategy's PeriodicEval
    # because we are going to use ParallelEval to evaluate the model.
    loggers = [InteractiveLogger()] if use_logger else []
    evaluation_plugin = EvaluationPlugin(accuracy_metrics(stream=True), loggers=loggers)

    strat = Naive(
        model=model,
        optimizer=optimizer,
        train_mb_size=128,
        train_epochs=1,
        eval_mb_size=128,
        device="cuda",
        peval_mode="iteration",
        plugins=plugins,
        evaluator=evaluation_plugin,
        **kwargs,
    )
    return strat


def test_speed():
    ray.init(num_cpus=24, num_gpus=1)
    scenario = SplitMNIST(5)
    scenario = benchmark_with_validation_stream(scenario, validation_size=0.05)
    model = SimpleMLP(10)

    ###########################
    #  Non parallel eval Time #
    ###########################

    plugins = []

    strat = create_strategy(
        copy.deepcopy(model), plugins, use_logger=True, eval_every=-1
    )
    time_a = time.time()
    batch_streams = scenario.streams.values()
    for t, experience in enumerate(scenario.train_stream):
        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=128,
            access_task_boundaries=False,
        )
        train_stream = ocl_scenario.train_stream
        strat.train(
            train_stream, eval_streams=[scenario.valid_stream[: t + 1]], num_workers=0
        )
    time_b = time.time()
    strat.eval(scenario.test_stream)

    print(f"Time spent without eval: {time_b - time_a}")

    ########################
    #  ParallelEval Time   #
    ########################

    plugins = [
        ParallelEvaluationPlugin(
            metrics=accuracy_metrics(stream=True),
            eval_every=1,
            num_actors=4,
            eval_mb_size=128,
            max_launched=100,
            loggers=[JSONLogger("./results/logs.json")],
        )
    ]

    strat = create_strategy(copy.deepcopy(model), plugins, eval_every=-1)
    time_a = time.time()
    batch_streams = scenario.streams.values()
    for t, experience in enumerate(scenario.train_stream):
        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=128,
            access_task_boundaries=False,
        )
        train_stream = ocl_scenario.train_stream
        strat.train(
            train_stream, eval_streams=[scenario.valid_stream[: t + 1]], num_workers=0
        )

    strat.eval(scenario.test_stream)
    # Have to call that otherwise it's cheating
    # this is otherwise called at exit so would be run in
    # parallel with the next test
    ray.get(plugins[0].scheduler.scheduled_tasks)
    plugins[0].write_actor_logs()
    time_b = time.time()

    print(f"Time spent by parallel eval: {time_b - time_a}")

    ###########################
    #  Non parallel eval Time #
    ###########################

    plugins = []

    strat = create_strategy(copy.deepcopy(model), plugins, eval_every=1)
    time_a = time.time()
    batch_streams = scenario.streams.values()
    for t, experience in enumerate(scenario.train_stream):
        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=128,
            access_task_boundaries=False,
        )
        train_stream = ocl_scenario.train_stream
        strat.train(
            train_stream, eval_streams=[scenario.valid_stream[: t + 1]], num_workers=0
        )
    time_b = time.time()
    strat.eval(scenario.test_stream)

    print(f"Time spent by non parallel eval: {time_b - time_a}")


test_speed()
