import os
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from avalanche.benchmarks.scenarios import OnlineCLExperience
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from src.toolkit.utils import get_last_layer_name


class FeatureExtractorModel(torch.nn.Module):
    def __init__(self, feature_extractor, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)


class SKLearnLayer(nn.Module):
    def __init__(self, sklearn_classifier):
        super().__init__()
        self.classifier = sklearn_classifier
        self.scaler = preprocessing.StandardScaler()

    def forward(self, x):
        device = x.device
        x = x.cpu().numpy()
        x = self.scaler.transform(x)
        return torch.tensor(self.classifier.predict_proba(x)).to(device)

    def fit(self, x, y, **kwargs):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        self.classifier.fit(x, y, **kwargs)


class SKLearnProbingPlugin(SupervisedPlugin):
    """
    Probing the already existing models by tuning
    the task-specific classifiers with the current dataset
    """

    def __init__(self, models_dir, prefix="model"):
        super().__init__()
        self.model_dir = models_dir
        self.prefix = prefix

        self.initial_model = None

    def before_training(self, strategy, **kwargs):
        if self.initial_model is None:
            self.last_layer_name, _ = get_last_layer_name(strategy.model)
            self.initial_model = copy.deepcopy(strategy.model)
            pass

    def before_training_epoch(self, strategy, **kwargs):
        strategy.stop_training()

        # Skip strategy training and
        # train the sklearn classifier

        features = []
        targets = []

        with torch.no_grad():
            for i in range(3):
                for strategy.mbatch in strategy.dataloader:
                    strategy._unpack_minibatch()

                    strategy._before_forward(**kwargs)
                    strategy.mb_output = strategy.model.feature_extractor(strategy.mb_x)
                    strategy._after_forward(**kwargs)

                    features.append(strategy.mb_output)
                    targets.append(strategy.mb_y)

        features = torch.cat(features)
        targets = torch.cat(targets)

        strategy.model.classifier.fit(features, targets)

    def before_training_exp(self, strategy, **kwargs):
        if isinstance(strategy.experience, OnlineCLExperience):
            raise ValueError("ProbingPlugin cannot be used on online experiences")

        # If no model dir is provided, keep the current model
        # ! First load state dict THEN reset head
        if self.model_dir is not None:
            # Here, set initial classifier as head
            strategy.model = copy.deepcopy(self.initial_model)
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

            # Now remove last layer and replace it with custom sklearn layer
            sklearn_classifier = LogisticRegression(max_iter=200)
            setattr(strategy.model, self.last_layer_name, nn.Identity())
            strategy.model = FeatureExtractorModel(
                strategy.model, SKLearnLayer(sklearn_classifier)
            )

        # Freeze the whole model except last classification layer
        for p in strategy.model.parameters():
            p.requires_grad = False
            p.grad = None

        strategy.model.eval()
