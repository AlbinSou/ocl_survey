import copy
import os
from collections import defaultdict

import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from avalanche.benchmarks.scenarios import OnlineCLExperience
from avalanche.models import SCRModel
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from src.factories.benchmark_factory import DS_CLASSES
from src.toolkit.utils import get_last_layer_name


def default_loading(model, state_dict, last_layer_name, replace=True, strict=True):
    model.load_state_dict(state_dict, strict=strict)

    if replace:
        # Now remove last layer and replace it with custom sklearn layer
        sklearn_classifier = LogisticRegression(max_iter=200)
        setattr(model, last_layer_name, nn.Identity())
        model = FeatureExtractorModel(
            model, SKLearnLayer(sklearn_classifier)
        )

    return model


def loading_der(model, state_dict, last_layer_name):
    # DER sets from the start a linear layer with all the classes
    last_layer_name, in_features = get_last_layer_name(model)
    success = False
    for name, n_classes in DS_CLASSES.items():
        try:
            setattr(model, last_layer_name, nn.Linear(in_features, n_classes))
            model = default_loading(model, state_dict, last_layer_name, replace=True)
            success = True
            break
        except Exception as e:
            print(f"Failed to load with der strategy: {e}")
            success = False
            continue
    if not success:
        return None

    return model


def loading_scr(model, state_dict, last_layer_name):
    last_layer_name, in_features = get_last_layer_name(model)
    setattr(model, last_layer_name, torch.nn.Identity())
    projection_network = torch.nn.Sequential(
        torch.nn.Linear(in_features, in_features),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(in_features, 128),
    )
    # a NCM Classifier is used at eval time
    model = SCRModel(feature_extractor=model, projection=projection_network)
    model = default_loading(model, state_dict, last_layer_name, replace=False, strict=False)

    sklearn_classifier = LogisticRegression(max_iter=200)
    setattr(model, "eval_classifier", nn.Identity())
    setattr(model, "train_classifier", nn.Identity())
    model = FeatureExtractorModel(
        model, SKLearnLayer(sklearn_classifier)
    )
    return model


loading_methods = [default_loading, loading_der, loading_scr]

def load_model(model, state_dict, last_layer_name):
    model = copy.deepcopy(model)
    counter = 0
    for method in loading_methods:
        counter += 1
        try:
            new_model = method(model, state_dict, last_layer_name)
            if new_model is not None:
                break
        except Exception as e:
            new_model = None
            continue
    if new_model is not None:
        print(f"Successfully loaded with method {counter}")
    else:
        raise Exception("Failed to load model")
    return new_model


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

            state_dict = torch.load(
                os.path.join(
                    self.model_dir,
                    f"{self.prefix}_{strategy.experience.current_experience}.ckpt",
                ),
                map_location=strategy.device,
            )

            strategy.model = load_model(strategy.model, state_dict, self.last_layer_name)


            strategy.clock.train_iterations = strategy.experience.current_experience

        # Freeze the whole model except last classification layer
        for p in strategy.model.parameters():
            p.requires_grad = False
            p.grad = None

        strategy.model.eval()
