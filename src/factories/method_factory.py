#!/usr/bin/env python3

import os
from typing import List, Optional

import torch
import torch.nn as nn

import avalanche.logging as logging
import toolkit.utils as utils
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.plugins.evaluation import (EvaluationPlugin,
                                                   default_evaluator)
from avalanche.training.supervised import DER, Naive
from toolkit.json_logger import JSONLogger
from toolkit.parallel_eval import ParallelEvaluationPlugin


"""
Method Factory
"""


def create_strategy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    plugins: Optional[List[SupervisedPlugin]] = None,
    logdir: str = None,
    name: str = None,
    evaluation_kwargs=None,
    strategy_kwargs=None,
):
    strategy_dict = {
        "model": model,
        "optimizer": optimizer,
        "criterion": nn.CrossEntropyLoss(),
    }
    strategy_args = utils.extract_kwargs(
        ["train_mb_size", "train_epochs", "eval_mb_size", "device"], strategy_kwargs
    )
    strategy_dict.update(strategy_args)
    strategy_eval_args = utils.extract_kwargs(
        ["eval_every", "peval_mode"], evaluation_kwargs
    )
    strategy_dict.update(strategy_eval_args)

    evaluator, parallel_eval_plugin = create_evaluator(
        logdir=logdir, **evaluation_kwargs
    )

    # When using parallel eval
    # let it do the job of Peval
    if parallel_eval_plugin is not None:
        strategy_dict["eval_every"] = -1

    if name == "der":
        strategy = "DER"
        model.linear = nn.Linear(model.linear.classifier.in_features, 100)
        der_args = utils.extract_kwargs(
            ["alpha", "beta", "batch_size_mem", "mem_size"], strategy_kwargs
        )

    cl_strategy = globals()[strategy](**strategy_dict)

    return cl_strategy


def get_loggers(loggers_list, logdir, prefix="logs"):
    loggers = []

    if loggers_list is None:
        return loggers

    for logger in loggers_list:
        if logger == "interactive":
            loggers.append(logging.InteractiveLogger())
        if logger == "tensorboard":
            loggers.append(logging.TensorboardLogger(logdir))
        if logger == "text":
            loggers.append(logging.TextLogger(os.path.join(logdir, f"{prefix}.txt")))
        if logger == "json":
            loggers.append(
                JSONLogger(os.path.join(logdir, f"{prefix}.json"), autoupdate=False)
            )
    return loggers


def get_metrics(metric_names):
    metrics = []
    for m in metric_names:
        metrics.append(globals()[m](stream=True))
    return metrics


def create_evaluator(
    metrics,
    logdir,
    loggers_strategy=None,
    loggers_parallel=None,
    parallel_evaluation=False,
    **parallel_eval_kwargs,
):
    """
    If parallel evaluation is triggered, peval for strategy is turned to
    """
    strategy_metrics = get_metrics(metrics)
    loggers_strategy = get_loggers(loggers_strategy, logdir, prefix="logs")
    evaluator_strategy = EvaluationPlugin(strategy_metrics, loggers_strategy)

    parallel_eval_plugin = None
    if parallel_evaluation:
        loggers_parallel = get_loggers(
            loggers_parallel, logdir, prefix="logs_continual"
        )
        parallel_eval_plugin = ParallelEvaluationPlugin(
            metrics=metrics, loggers=loggers_parallel, **parallel_eval_kwargs
        )

    return evaluator_strategy, parallel_eval_plugin
