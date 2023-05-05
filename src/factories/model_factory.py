#!/usr/bin/env python3
import torch

import avalanche.models as models
import toolkit.utils as utils
from avalanche.training.plugins import LRSchedulerPlugin


def create_model(model_type: str):
    if model_type == "resnet18":
        model = models.SlimResNet18(1)
    return model


def get_optimizer(
    model,
    optimizer_type: str,
    kwargs_optimizer=None,
    scheduler_type: str = None,
    kwargs_scheduler=None,
):
    if optimizer_type == "SGD":
        sgd_args = ["lr", "momentum", "weight_decay"]
        sgd_args = utils.extract_kwargs(sgd_args, kwargs_optimizer)
        optimizer = torch.optim.SGD(model.parameters(), **sgd_args)
    elif optimizer_type == "Adam":
        adam_args = ["lr", "weight_decay"]
        adam_args = utils.extract_kwargs(adam_args, kwargs_optimizer)
        optimizer = torch.optim.Adam(model.parameters(), **adam_args)

    assert optimizer is not None

    # Scheduler
    scheduler = None
    if scheduler_type == "step":
        assert kwargs_scheduler is not None
        scheduler_args = ["lr", "weight_decay"]
        scheduler_args = utils.extract_kwargs(scheduler_args, kwargs_scheduler)
        scheduler = torch.optim.lr_scheduler.StepLR(**scheduler_args)

    if scheduler is not None:
        plugin_args = []
        plugin_args = utils.extract_kwargs(plugin_args, kwargs_scheduler)
        scheduler_plugin = LRSchedulerPlugin(
            scheduler,
            reset_scheduler=False,
            step_granularity="iteration",
            **plugin_args
        )
    else:
        scheduler_plugin = None
    return optimizer, scheduler_plugin