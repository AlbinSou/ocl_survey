import os
from collections import defaultdict

import torch

from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class ProbingPlugin(SupervisedPlugin):
    """
    Probing the already existing models by tuning
    the task-specific classifiers with the current dataset
    """

    def __init__(self, models_dir, prefix="model", reset_last_layer=False):
        super().__init__()
        self.model_dir = models_dir
        self.prefix = prefix
        self.reset_last_layer = reset_last_layer

    def before_training_exp(self, strategy, **kwargs):
        # Load old model corresponding to current exp
        last_layer_name = list(strategy.model.named_parameters())[-1][0].split(".")[0]

        if self.reset_last_layer:
            in_features = getattr(strategy.model, last_layer_name).in_features
            setattr(
                strategy.model, last_layer_name, IncrementalClassifier(in_features, 1)
            )
            self.check_model_and_optimizer()

        # If no model dir is provided, keep the current model
        if self.model_dir is not None:
            strategy.model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.model_dir,
                        f"{self.prefix}_{strategy.experience.current_experience}.ckpt",
                    ),
                    map_location=strategy.device,
                )
            )
            for c in getattr(strategy.model, last_layer_name).modules():
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()

        # Freeze the whole model except last classification layer
        for p in strategy.model.parameters():
            p.requires_grad = False
            p.grad = None
        for p in getattr(strategy.model, last_layer_name).parameters():
            p.requires_grad = True

        strategy.model.eval()
