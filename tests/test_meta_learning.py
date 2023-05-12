#!/usr/bin/env python3
import copy
from typing import Iterable, List, Optional, Sequence, Union

import higher
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

import src.factories.model_factory as model_factory
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.benchmarks.scenarios import OnlineCLScenario
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import OnlineSupervisedTemplate
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
        self.default_pg = copy.deepcopy(self.optimizer.param_groups[0])
        self.default_pg.pop("lr")
        self.default_pg.pop("params")
        self.make_optimizer()

    def make_optimizer(self, **kwargs):
        old_lrs = [group["lr"] for group in self.optimizer.param_groups]
        if len(old_lrs) != len(list(self.model.parameters())):
            assert len(old_lrs) == 1
            old_lrs = [old_lrs[0] for p in self.model.parameters()]

        param_groups = [
            {
                "params": p,
                "lr": old_lrs[i],
                **self.default_pg,
            }
            for i, p in enumerate(self.model.parameters())
        ]

        self.optimizer = self.optimizer.__class__(param_groups)
        if self.lr_opt is None:
            self.learnable_lr = higher.optim.get_trainable_opt_params(
                self.optimizer, device=self.device
            )["lr"]
            self.lr_opt = torch.optim.Adam(self.learnable_lr)
        # TODO inherit momentum parameters from previous version

    def _train_exp(self, experience, eval_streams=None, **kwargs):
        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]

        self.meta_train_iter(**kwargs)
        self.train_iter(**kwargs)

    def meta_train_iter(self, **kwargs):
        if self.val_replay_loader is None:
            return

        current_experience_loader = cycle(
            torch.utils.data.DataLoader(
                self.adapted_dataset,
                batch_size=self.train_mb_size,
                shuffle=True,
            )
        )

        self.model.train()
        self.lr_opt.zero_grad()
        with higher.innerloop_ctx(
            self.model,
            self.optimizer,
            copy_initial_weights=True,
            track_higher_grads=True,
            device=self.device,
            override={"lr": self.learnable_lr},
        ) as (fmodel, diffopt):
            train_loss = 0.0

            fmodel.train()
            for _ in range(self.train_epochs):
                self._before_training_epoch(**kwargs)

                if self._stop_training:  # Early stopping
                    self._stop_training = False
                    break

                self.meta_training_epoch(fmodel, diffopt, train_loss, **kwargs)
                self._after_training_epoch(**kwargs)

            test_loss = 0

            fmodel.eval()
            for i in range(self.meta_val_steps):
                self.mbatch = next(current_experience_loader)
                self._unpack_minibatch()

                val_batch = next(self.val_replay_loader)
                self._concat_batch(val_batch)

                self.loss = 0.0

                self.mb_output = fmodel(self.mb_x)

                # Loss & Backward
                self.loss += F.cross_entropy(self.mb_output, self.mb_y)
                test_loss += self.loss

            test_loss += (
                1.0
                * sum([torch.square(lr - 0.1) for lr in self.learnable_lr])
                / len(self.learnable_lr)
            )
            test_loss.backward()

        self.lr_opt.step()

        # Now set new learning rates
        for lr in self.learnable_lr:
            lr.data.clamp_min_(0.001)

        higher.optim.apply_trainable_opt_params(
            self.optimizer, {"lr": self.learnable_lr}
        )

        self._log_lr()

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

    def meta_training_epoch(self, fmodel, diffopt, train_loss, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
            self.loss = 0.0
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = fmodel(self.mb_x)
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += F.cross_entropy(self.mb_output, self.mb_y)

            self._before_backward(**kwargs)
            self._after_backward(**kwargs)

            train_loss += self.loss

            # Optimization step
            diffopt.step(train_loss)

            self._after_training_iteration(**kwargs)

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

    def _log_lr(self):
        for log in self.evaluator.loggers:
            if isinstance(log, TensorboardLogger):
                tb_logger = log
        log.writer.add_scalars(
            "learning_rates",
            {
                f"lr_{i}": group["lr"]
                for i, group in enumerate(self.optimizer.param_groups)
            },
            global_step=self.clock.train_iterations,
        )


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
        },
    )
    scenario = SplitCIFAR10(
        5,
        first_exp_with_half_classes=False,
        return_task_id=False,
        seed=0,
        fixed_class_order=None,
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
        dataset_root="/DATA/data",
    )

    # scenario = benchmark_with_validation_stream(scenario, validation_size=0.05)

    model = model_factory.create_model("resnet18")
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    evaluator = EvaluationPlugin(
        accuracy_metrics(stream=True),
        loggers=[InteractiveLogger(), TensorboardLogger("./results")],
    )

    strategy = OnlineMetaSGD(
        model,
        optimizer,
        buffer_size=args.mem_size,
        buffer_val_size=args.mem_size,
        evaluator=evaluator,
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
