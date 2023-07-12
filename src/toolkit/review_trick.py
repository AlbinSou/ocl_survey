#!/usr/bin/env python3
import copy
import os
from collections import defaultdict

import torch
import torch.nn as nn

from avalanche.benchmarks.scenarios import OnlineCLExperience
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class ReviewTrickPlugin(SupervisedPlugin):
    def __init__(self, storage_policy, num_epochs=10):
        self.storage_policy = storage_policy
        self.num_epochs = num_epochs

    @torch.enable_grad()
    def before_eval(self, strategy, **kwargs):
        memory_loader = torch.utils.data.DataLoader(
            self.storage_policy.buffer,
            shuffle=True,
            batch_size=strategy.train_mb_size * 6,
        )

        model_copy = copy.deepcopy(strategy.model)
        self.training_model = strategy.model
        strategy.model = model_copy
        strategy.model.train()

        optimizer = torch.optim.SGD(
            strategy.model.parameters(), lr=strategy.optimizer.param_groups[0]["lr"]
        )

        for epoch in range(self.num_epochs):
            for strategy.mbatch in memory_loader:
                strategy._unpack_minibatch()
                strategy.mb_output = strategy.forward()
                loss = strategy.criterion()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        strategy.model.eval()

    def after_eval(self, strategy, **kwargs):
        strategy.model = self.training_model
