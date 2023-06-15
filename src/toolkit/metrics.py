#!/usr/bin/env python3
import torch
from torch import Tensor
from collections import defaultdict
from avalanche.training.plugins import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue

class ClockLoggingPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()
    
    def before_eval(self, strategy, **kwargs):
        strategy.evaluator.publish_metric_value(
            MetricValue(
                "Metric",
                "mb_index",
                strategy.clock.train_exp_counter,
                x_plot=strategy.clock.train_iterations,
            )
        )
