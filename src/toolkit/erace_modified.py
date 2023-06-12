#!/usr/bin/env python3
import copy
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils import concat_datasets
from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.regularization import RegularizationMethod
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate


def cross_entropy_with_oh_targets(outputs, targets, eps=1e-5):
    """Calculates cross-entropy with temperature scaling,
    targets can also be soft targets but they must sum to 1"""
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    ce = -(targets * outputs.log()).sum(1)
    ce = ce.mean()
    return ce


class ACECriterion(RegularizationMethod):
    """
    Asymetric cross-entropy (ACE) Criterion used in
    "New Insights on Reducing Abrupt Representation
    Change in Online Continual Learning"
    by Lucas Caccia et. al.
    https://openreview.net/forum?id=N8MaByOzUfb
    """

    def __init__(self):
        pass

    def __call__(
        self,
        out_in,
        target_in,
        out_buffer,
        target_buffer,
        weight_current=0.5,
        weight_buffer=0.5,
    ):
        current_classes = torch.unique(target_in)
        loss_buffer = F.cross_entropy(out_buffer, target_buffer)
        oh_target_in = F.one_hot(target_in, num_classes=out_in.shape[1])
        oh_target_in = oh_target_in[:, current_classes]
        loss_current = cross_entropy_with_oh_targets(
            out_in[:, current_classes], oh_target_in
        )
        return (weight_buffer * loss_buffer + weight_current * loss_current)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


class ER_ACE(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        batch_size_mem: int,
        criterion=CrossEntropyLoss(),
        mem_size: int = 200,
        alpha: float = 1.0,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
        peval_mode="experience",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int : Fixed memory size
        :param alpha: float : Weight applied to the loss on current data (default=0.5)
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None
        self.ace_criterion = ACECriterion()
        self.alpha = alpha

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            if self.replay_loader is not None:
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = next(
                    self.replay_loader
                )
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = (
                    self.mb_buffer_x.to(self.device),
                    self.mb_buffer_y.to(self.device),
                    self.mb_buffer_tid.to(self.device),
                )

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            if self.replay_loader is not None:
                self.mb_buffer_out = avalanche_forward(
                    self.model, self.mb_buffer_x, self.mb_buffer_tid
                )
            self._after_forward(**kwargs)

            # Loss & Backward
            if self.replay_loader is None:
                self.loss += self.criterion()
            else:
                self.loss += self.ace_criterion(
                    self.mb_output,
                    self.mb_y,
                    self.mb_buffer_out,
                    self.mb_buffer_y,
                    weight_current=self.alpha,
                    weight_buffer=(1 - self.alpha),
                )

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _before_training_exp(self, **kwargs):
        self.storage_policy.update(self, **kwargs)
        # Take all classes for ER ACE loss
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                )
            )
        else:
            self.replay_loader = None

        super()._before_training_exp(**kwargs)

    def _train_cleanup(self):
        super()._train_cleanup()
        # reset the value to avoid serialization failures
        self.replay_loader = None
