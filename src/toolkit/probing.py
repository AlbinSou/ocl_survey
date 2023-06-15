import os
from collections import defaultdict

import torch
import torch.nn as nn

from avalanche.benchmarks.scenarios import OnlineCLExperience
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

        self.initial_classifier = None

    def before_training(self, strategy, **kwargs):
        if self.initial_classifier is None:
            # Load old model corresponding to current exp
            self.last_layer_name = list(strategy.model.named_parameters())[-1][0].split(
                "."
            )[0]
            self.initial_classifier = getattr(strategy.model, self.last_layer_name)

    def before_training_exp(self, strategy, **kwargs):
        if isinstance(strategy.experience, OnlineCLExperience):
            raise ValueError("ProbingPlugin cannot be used on online experiences")

        # If no model dir is provided, keep the current model
        # ! First load state dict THEN reset head
        if self.model_dir is not None:
            # Here, set initial classifier as head
            setattr(strategy.model, self.last_layer_name, self.initial_classifier)
            strategy.model = strategy.model_adaptation()

            strategy.model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.model_dir,
                        f"{self.prefix}_{strategy.experience.current_experience}.ckpt",
                    ),
                    map_location=strategy.device,
                )
            )
            for c in getattr(strategy.model, self.last_layer_name).modules():
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()

        if self.reset_last_layer:
            last_layer = getattr(strategy.model, self.last_layer_name)

            if hasattr(last_layer, "in_features"):
                in_features = last_layer.in_features
            else:
                in_features = last_layer.classifier.in_features

            setattr(
                strategy.model,
                self.last_layer_name,
                # nn.Linear(in_features, self.initial_classifier.classifier.out_features),
                IncrementalClassifier(
                    in_features,
                    self.initial_classifier.classifier.out_features,
                    masking=False,
                ),
            )
            strategy.check_model_and_optimizer()

        # Freeze the whole model except last classification layer
        for p in strategy.model.parameters():
            p.requires_grad = False
            p.grad = None
        for p in getattr(strategy.model, self.last_layer_name).parameters():
            p.requires_grad = True

        strategy.model.eval()
