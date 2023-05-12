###############################################################
# Text experiment for meta-learning the learning rate
# This test assume a single learning rate for the whole model.
###############################################################

#!/usr/bin/env python3
import sys
sys.path.append(".")


from typing import Iterable, List, Optional
import copy
import torch
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F
import higher
from avalanche.training.templates import OnlineSupervisedTemplate
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.logging import (
    InteractiveLogger, TensorboardLogger, WandBLogger)
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.scenarios import OnlineCLScenario
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100

from avalanche.evaluation import GenericPluginMetric
from avalanche.evaluation.metric_results import MetricValue

import src.factories.model_factory as model_factory
from toolkit.utils import create_default_args


def cycle(loader):
    while True:
        for mbatch in loader:
            yield mbatch


class OnlineMetaSGD(OnlineSupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        buffer_size=500,
        buffer_val_size=500,
        meta_val_steps=20,
        lr_meta_lr=0.03,
        fixed_init_meta_lr=False,
        train_passes: int = 1,
        train_mb_size: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
        peval_mode="iteration",
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_passes=train_passes,
            train_mb_size=train_mb_size,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )
        self.replay_buffer = ClassBalancedBuffer(buffer_size)
        self.val_replay_buffer = ClassBalancedBuffer(buffer_val_size)
        self.replay_loader = None
        self.val_replay_loader = None
        self.lr_opt = None
        self.meta_val_steps = meta_val_steps
        self.replay_mask = None

        self.lr_meta_lr = lr_meta_lr
        self.fixed_init_meta_lr = fixed_init_meta_lr
        self.init_lr = self.optimizer.param_groups[0]["lr"]

    def _train_exp(self, experience, eval_streams=None, **kwargs):
        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]

        self.meta_update_lr(**kwargs)
        self.train_iter(**kwargs)

    def train_iter(self, **kwargs):
        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break
            self.training_epoch()
            self._after_training_epoch(**kwargs)

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp()
        self.replay_buffer.update(self)
        self.val_replay_buffer.update(self)

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp()
        if len(self.replay_buffer.buffer) > self.train_mb_size:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    self.replay_buffer.buffer,
                    batch_size=self.train_mb_size,
                    shuffle=True,
                    drop_last=True,
                )
            )
            self.val_replay_loader = cycle(
                torch.utils.data.DataLoader(
                    self.val_replay_buffer.buffer,
                    batch_size=self.train_mb_size,
                    shuffle=True,
                    drop_last=True,
                )
            )

    def meta_update_lr(self, **kwargs):
        if self.val_replay_loader is None:
            return
        
        if self.lr_meta_lr == 0:
            return
        
        # Initialize meta-lr as a parameter
        if self.fixed_init_meta_lr:
            meta_lr = torch.nn.Parameter(torch.tensor(self.fixed_init_meta_lr),
                                         requires_grad=True)
        else:
            current_lr = self.optimizer.param_groups[0]["lr"]
            meta_lr = torch.nn.Parameter(torch.tensor(current_lr),
                                         requires_grad=True)

        # Fast parameters for the current model
        fast_params_0 = {n: copy.deepcopy(p) for (n, p) in
                         model.named_parameters()}

        # Fast update with the current batch:
        x, y, _ = next(iter(self.dataloader))
        x, y = x.to(self.device), y.to(self.device)

        logits_fast_1 = torch.func.functional_call(model, fast_params_0, (x,))
        loss = F.cross_entropy(logits_fast_1, y)

        # Compute gradient with respect to the current fast weights
        grads_inner_1 = list(torch.autograd.grad(loss, fast_params_0.values(),
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 allow_unused=True,))

        # New fast parameters
        fast_params_1 = {n: p - meta_lr * g if g is not None else p
                         for ((n, p), g) in
                         zip(fast_params_0.items(),  grads_inner_1)}

        # Meta-loss
        x_b, y_b, _ = next(self.val_replay_loader)
        x_b, y_b = x_b.to(self.device), y_b.to(self.device)

        x_mixed = torch.cat((x_b, x), dim=0)
        y_mixed = torch.cat((y_b, y), dim=0)

        logits_meta = torch.func.functional_call(
            model, fast_params_1, (x_mixed,))

        meta_loss = F.cross_entropy(logits_meta, y_mixed)
        meta_loss.backward()

        meta_lr = meta_lr - self.lr_meta_lr * meta_lr.grad

        new_lr = meta_lr.data.item()
        new_lr = max(0.0001, new_lr)

        # Update optimizer LRs
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return meta_lr

    def _concat_batch(self, mreplay):
        replay_x, replay_y, replay_tid = (
            mreplay[0].to(self.device),
            mreplay[1].to(self.device),
            mreplay[2].to(self.device),
        )

        current_mask = torch.zeros(len(self.mbatch[0])).to(self.device)
        self.mbatch[0] = torch.cat((replay_x, self.mbatch[0]), dim=0)
        self.mbatch[1] = torch.cat((replay_y, self.mbatch[1]), dim=0)
        self.mbatch[2] = torch.cat((replay_tid, self.mbatch[2]), dim=0)

        replay_mask = torch.ones(len(replay_x)).to(self.device)
        self.replay_mask = torch.cat((replay_mask, current_mask), dim=0)

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

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)

            # Add replay batch
            if self.replay_loader is not None:
                mreplay = next(self.replay_loader)
                old_len = len(self.mb_x)
                self._concat_batch(mreplay)
                assert len(self.mb_x) == old_len + len(mreplay[0])

            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)


class LRMetric(GenericPluginMetric[float]):
    def __init__(self):
        super().__init__(
            None, reset_at="experience", emit_at="experience", 
            mode="train"
        )
        self.current_lr = None

    def result(self, strategy) -> float:
        return self.current_lr

    def reset(self, strategy) -> None:
        return
    
    def after_training_exp(self, strategy) -> None:
        self.current_lr = strategy.optimizer.param_groups[0]["lr"]
        metric_value = self.current_lr
        plot_x_position = strategy.clock.train_iterations
        metric_name = "Learning_Rate"
        metric = [MetricValue(self, metric_name, metric_value, plot_x_position)]

        return metric

    def __str__(self):
        return "Learning_Rate"


if __name__ == "__main__":
    args = create_default_args(
        {
            "cuda": 0,
            "mem_size": 500,
            "lr": 0.1,
            "train_mb_size": 64,
            "meta_val_steps": 3,
            "train_passes": 3,
            "seed": 0,
            "lr_meta_lr": 0.0,
            "fixed_init_meta_lr": False,
        },
    )

    wandb_proj = "tests"

    scenario = SplitCIFAR10(
        5,
        first_exp_with_half_classes=False,
        return_task_id=False,
        seed=0,
        fixed_class_order=None,
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
        dataset_root="./data/datasets/",
    )

    # scenario = benchmark_with_validation_stream(scenario, validation_size=0.05)

    model = model_factory.create_model("resnet18")
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    lr_metric = LRMetric()
    loggers = [InteractiveLogger(), TensorboardLogger("./results")]
    if wandb_proj is not None:
        loggers.append(WandBLogger(project_name=wandb_proj, 
                                   run_name=None,
                                   config=vars(args)))
    
    metrics = accuracy_metrics(stream=True) + [lr_metric]
    evaluator = EvaluationPlugin(
        metrics,
        loggers=loggers,
    )

    device = torch.device("mps")

    strategy = OnlineMetaSGD(
        model,
        optimizer,
        buffer_size=args.mem_size,
        buffer_val_size=args.mem_size,
        evaluator=evaluator,
        device=device,
        lr_meta_lr=args.lr_meta_lr,
        fixed_init_meta_lr=args.fixed_init_meta_lr,
        train_mb_size=args.train_mb_size,
        eval_mb_size=args.train_mb_size,
        train_passes=args.train_passes,
        meta_val_steps=args.meta_val_steps,
        eval_every=-1,
        peval_mode="experience",
    )

    batch_streams = scenario.streams.values()
    for t, experience in enumerate(scenario.train_stream):
        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=args.train_mb_size,
            access_task_boundaries=True,
        )

        strategy.train(
            ocl_scenario.train_stream,
            # eval_streams=[scenario.valid_stream[: t + 1]],
            eval_streams=[],
            num_workers=0,
            drop_last=True,
        )

        strategy.eval(scenario.test_stream[: t + 1])
