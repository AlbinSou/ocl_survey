import copy
import itertools
from math import ceil
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch.nn import BCELoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import (classification_subset,
                                        make_tensor_classification_dataset)
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.models import NCMClassifier, TrainEvalModel
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.utils import at_task_boundary


class OnlineICaRLLossPlugin(SupervisedPlugin):
    """
    ICaRLLossPlugin
    Similar to the Knowledge Distillation Loss. Works as follows:
        The target is constructed by taking the one-hot vector target for the
        current sample and assigning to the position corresponding to the
        past classes the output of the old model on the current sample.
        Doesn't work if classes observed in previous experiences might be
        observed again in future training experiences.
    """

    def __init__(self):
        super().__init__()
        self.criterion = BCELoss()

        self.old_classes = []
        self.old_model = None
        self.old_logits = None

    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            with torch.no_grad():
                self.old_logits = self.old_model(strategy.mb_x)

    def __call__(self, logits, targets):
        predictions = torch.sigmoid(logits)

        one_hot = torch.zeros(
            targets.shape[0],
            logits.shape[1],
            dtype=torch.float,
            device=logits.device,
        )
        one_hot[range(len(targets)), targets.long()] = 1

        if self.old_logits is not None:
            old_predictions = torch.sigmoid(self.old_logits)
            one_hot[:, self.old_classes] = old_predictions[:, self.old_classes]
            self.old_logits = None

        return self.criterion(predictions, one_hot)

    def after_training_exp(self, strategy, **kwargs):
        if not at_task_boundary(strategy.experience):
            return

        if self.old_model is None:
            old_model = copy.deepcopy(strategy.model)
            self.old_model = old_model.to(strategy.device)

        self.old_model.load_state_dict(strategy.model.state_dict())

        self.old_classes += np.unique(strategy.experience.dataset.targets).tolist()


class OnlineICaRL(SupervisedTemplate):
    """iCaRL Strategy.

    This strategy does not use task identities.

    Modified to not use Herding
    """

    def __init__(
        self,
        feature_extractor: Module,
        classifier: Module,
        optimizer: Optimizer,
        mem_size: int = 200,
        batch_size_mem: int = None,
        criterion=OnlineICaRLLossPlugin(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
    ):
        """Init.

        :param feature_extractor: The feature extractor.
        :param classifier: The differentiable classifier that takes as input
            the output of the feature extractor.
        :param optimizer: The optimizer to use.
        :param mem_size: The nuber of patterns saved in the memory.
        :param buffer_transform: transform applied on buffer elements already
            modified by test_transform (if specified) before being used for
            replay
        :param fixed_memory: If True a memory of size memory_size is
            allocated and partitioned between samples from the observed
            experiences. If False every time a new class is observed
            memory_size samples of that class are added to the memory.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """
        model = TrainEvalModel(
            feature_extractor,
            train_classifier=classifier,
            eval_classifier=NCMClassifier(normalize=True),
        )

        storage_policy = ClassBalancedBuffer(
            mem_size,
            adaptive_size=True,
        )

        replay_plugin = ReplayPlugin(
            mem_size,
            batch_size=train_mb_size,
            batch_size_mem=batch_size_mem,
            storage_policy=storage_policy,
        )

        icarl = _ICaRLPlugin(replay_plugin)

        if plugins is None:
            plugins = [icarl, replay_plugin]
        else:
            plugins += [icarl, replay_plugin]

        if isinstance(criterion, SupervisedPlugin):
            plugins += [criterion]

        super().__init__(
            model,
            optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )


class _ICaRLPlugin(SupervisedPlugin):
    """
    iCaRL Plugin.
    iCaRL uses nearest class exemplar classification to prevent
    forgetting to occur at the classification layer. The feature extractor
    is continually learned using replay and distillation. The exemplars
    used for replay and classification are selected through herding.
    This plugin does not use task identities.
    """

    def __init__(self, replay_plugin):
        """
        :param memory_size: amount of patterns saved in the memory.
        :param buffer_transform: transform applied on buffer elements already
            modified by test_transform (if specified) before being used for
             replay
        :param fixed_memory: If True a memory of size memory_size is
            allocated and partitioned between samples from the observed
            experiences. If False every time a new class is observed
            memory_size samples of that class are added to the memory.
        """
        super().__init__()

        self.replay_plugin = replay_plugin

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.eval()
        self.compute_class_means(strategy)
        strategy.model.train()

    @torch.no_grad()
    def compute_class_means(self, strategy):
        per_class_embeddings = {}

        buffer_loader = DataLoader(
            self.replay_plugin.storage_policy.buffer, 
            batch_size=strategy.eval_mb_size
        )

        for batch_x, batch_y, batch_tid in buffer_loader:
            batch_x = batch_x.to(strategy.device)
            out_features = strategy.model.feature_extractor(batch_x)
            for class_id in torch.unique(batch_y):
                id_select = batch_y == class_id
                class_features = out_features[id_select]

                if class_id not in per_class_embeddings:
                    per_class_embeddings[int(class_id)] = {
                        "sum": torch.sum(class_features, dim=0),
                        "total": len(class_features),
                    }
                else:
                    per_class_embeddings[int(class_id)]["sum"] += torch.sum(class_features, dim=0)
                    per_class_embeddings[int(class_id)]["total"] += len(class_features)

        class_means = {}
        for class_id in per_class_embeddings:
            # Should we normalize somewhere ?
            class_means[class_id] = (
                per_class_embeddings[class_id]["sum"]
                / per_class_embeddings[class_id]["total"]
            )
    
        if len(class_means) > 0:
            strategy.model.eval_classifier.replace_class_means_dict(class_means)
