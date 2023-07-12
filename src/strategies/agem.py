import random
import warnings
from typing import Any, Iterator, List, Optional

import torch
from torch import Tensor

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.utils import cycle


class AGEMPlugin(SupervisedPlugin):
    """Average Gradient Episodic Memory Plugin.

    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, mem_size: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__()

        self.sample_size = int(sample_size)

        self.storage_policy = ClassBalancedBuffer(max_size=mem_size, adaptive_size=True)
        self.buffer_dliter = None

        # Placeholder Tensor to avoid typing issues
        self.reference_gradients: Tensor = torch.empty(0)

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """

        if self.buffer_dliter is not None:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()
            xref, yref, tid = mb[0], mb[1], mb[-1]
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            loss = strategy._criterion(out, yref)
            loss.backward()
            # gradient can be None for some head on multi-headed models
            reference_gradients_list = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            self.reference_gradients = torch.cat(reference_gradients_list)
            strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if self.buffer_dliter is not None:
            current_gradients_list = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients_list)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(
                    self.reference_gradients, self.reference_gradients
                )
                grad_proj = current_gradients - self.reference_gradients * alpha2

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count : count + n_param].view_as(p))
                    count += n_param

    def after_training_exp(self, strategy, **kwargs):
        """Update replay memory with patterns from current experience."""
        self.storage_policy.update_from_dataset(strategy.experience.dataset)
        self.buffer_dliter = torch.utils.data.DataLoader(
            self.storage_policy.buffer, shuffle=True, batch_size=self.sample_size
        )
        self.buffer_dliter = cycle(self.buffer_dliter)

    def sample_from_memory(self):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        return next(self.buffer_dliter)
