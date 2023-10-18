import argparse
import copy

import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.benchmarks.datasets.dataset_utils import \
    default_dataset_location
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metrics import (accuracy_metrics, forgetting_metrics,
                                          loss_metrics)
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.models import IncrementalClassifier, SlimResNet18
from avalanche.training.plugins import (EvaluationPlugin, ReplayPlugin,
                                        SupervisedPlugin)
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.supervised import Naive
from torchvision import transforms

from src.strategies.lwf import LwFPlugin
from src.toolkit.utils import clear_tensorboard_files

###############


def scheduler_function(start, coeff, it, threshold=1000, end=3.0):
    if it > threshold:
        if start + coeff * (it - threshold) < end:
            return start + coeff * (it - threshold)
        else:
            return end
    else:
        return start


# Insert code for the demo

###############

class LambdaScheduler(SupervisedPlugin):
    """
    Schedules a plugin parameter by applying a linear ramp on
    """

    def __init__(self, key, plugin, start, coeff=0.001):
        self.key = key
        self.plugin = plugin
        self.start = start
        self.coeff = coeff

    def after_training_exp(self, strategy, **kwargs):
        # Online setting so one exp = one mini batch
        # TODO
        pass

###############


def main(args):
    clear_tensorboard_files(".")

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    # ---------

    # ---------
    benchmark = SplitCIFAR10(
        n_experiences=5, seed=0, class_ids_from_zero_from_first_exp=True
    )
    # ---------

    # MODEL CREATION
    model = SlimResNet18(1)
    model.linear = IncrementalClassifier(model.linear.in_features)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    tensorboard_logger = TensorboardLogger(".")

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True),
        loss_metrics(stream=True),
        loggers=[interactive_logger, tensorboard_logger],
    )

    # CREATE THE STRATEGY INSTANCE (ONLINE-REPLAY)

    mem_size = 200
    storage_policy = ReservoirSamplingBuffer(max_size=mem_size)
    replay_plugin = ReplayPlugin(mem_size=mem_size, storage_policy=storage_policy)
    lwf_plugin = LwFPlugin(alpha=1, temperature=2)

    # ADD THE PLUGINS
    # TODO

    plugins = [replay_plugin, lwf_plugin]
    print(plugins)

    cl_strategy = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        nn.CrossEntropyLoss(),
        train_epochs=1,
        train_mb_size=10,
        eval_mb_size=64,
        device=device,
        evaluator=eval_plugin,
        plugins=plugins,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []

    # Create online benchmark
    batch_streams = benchmark.streams.values()
    for i, exp in enumerate(benchmark.train_stream):
        # Create online scenario from experience exp
        ocl_benchmark = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=exp,
            experience_size=10,
            access_task_boundaries=True,
        )
        # Train on the online train stream of the scenario
        cl_strategy.train(ocl_benchmark.train_stream)
        results.append(cl_strategy.eval(benchmark.test_stream[: i + 1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
