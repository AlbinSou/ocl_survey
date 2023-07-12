import atexit
import copy
import os
from typing import Dict, List, Tuple

import ray
import torch
from torch.optim import SGD

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from src.toolkit.json_logger import JSONLogger


class BlockingScheduler:
    def __init__(self, max_launched):
        self.max_launched = max_launched
        self.scheduled_tasks = []
        atexit.register(self.close)

    def schedule(self, function, *args, **kwargs):
        scheduled_task = None
        if len(self.scheduled_tasks) > self.max_launched:
            ready, _ = ray.wait(self.scheduled_tasks, num_returns=1)
            self.scheduled_tasks.remove(ready[0])
            scheduled_task = self.schedule(function, *args, **kwargs)
        else:
            scheduled_task = function.remote(*args, **kwargs)
            self.scheduled_tasks.append(scheduled_task)
        return scheduled_task

    def close(self):
        ray.get(self.scheduled_tasks)


# change this if you want more/less resource per worker
@ray.remote
class EvaluationActor(object):
    """Parallel Evaluation Actor.

    Methods called on different actors can execute in parallel, and methods
    called on the same actor are executed serially in the order that they
    are called. Methods on the same actor will share state with one another,
    as shown below."""

    def __init__(self, evaluator, **strat_args):
        """
        Constructor.
        :param strat_args: arguments of the strategy
        """

        self.strat = Naive(
            model=None,
            optimizer=None,
            evaluator=evaluator,
            **strat_args,
        )

    def eval(self, model, clock, stream, **kwargs):
        self.strat.model = model
        self.strat.clock = clock
        self.strat.eval(stream, **kwargs)
        self.write_files()

    def write_files(self):
        for logger in self.strat.evaluator.loggers:
            if isinstance(logger, JSONLogger):
                logger.update_json()


class ParallelEvaluationPlugin(SupervisedPlugin):
    """Schedules periodic evaluation during training."""

    def __init__(
        self,
        metrics,
        loggers,
        eval_every=-1,
        do_initial=False,
        num_actors=1,
        peval_mode="experience",
        max_launched=20,
        num_gpus=0.1,
        num_cpus=2,
        **actor_args
    ):
        """Init.

        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration', 'experience'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='experience').
        :param do_initial: whether to evaluate before each `train` call.
            Occasionally needed becuase some metrics need to know the
            accuracy before training.
        :param num_actors: number of evaluation actors to use. !CARE! if more than
            one actor is used, it will be unable to compute metrics that depend on
            several evaluation steps (like stability metrics).
        :param num_actors: number of evaluation actors to use. !CARE! if more than
            one actor is used, it will be unable to compute metrics that depend on
            several evaluation steps (like stability metrics).
        :param mode: evaluate after every experience or after
            every iteration (default: experience)
        :param max_launched: Maximum number of tasks scheduled to avoid OOM
        :param num_cpus: amount of cpus per actor
        :param num_gpus: amount of gpus per actor (can be float less than 1)
        :param **actor_args: parameters that will be given to the actor's strategy
        """
        super().__init__()
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.peval_mode = peval_mode
        assert peval_mode in ["iteration", "experience"]

        self.evaluator = EvaluationPlugin(
            *metrics,
            loggers=loggers,
        )

        self.eval_actors = self.create_actors(
            actor_args,
            num_actors,
        )

        self.eval_every = eval_every
        self.do_initial = do_initial and eval_every > -1
        self.eval_index = 0

        self.scheduler = BlockingScheduler(max_launched=max_launched)

    def after_training_exp(self, strategy, **kwargs):
        """Final eval after a learning experience."""
        self._peval(strategy, **kwargs)

    def before_training(self, strategy, **kwargs):
        self.stream_refs = []

        for el in strategy._eval_streams:
            self.stream_refs.append(ray.put(el))

        if self.do_initial:
            self._peval(strategy, **kwargs)

    def _peval(self, strategy, **kwargs):
        # you can run multiple actors in parallel here.
        clock_ref = ray.put(strategy.clock)
        for stream_ref in self.stream_refs:
            # Strategy model
            actor = self.eval_actors[self.eval_index % self.num_actors]

            self.scheduler.schedule(
                actor.eval,
                copy.deepcopy(strategy.model).cpu(),
                clock_ref,
                stream_ref,
                persistent_workers=True if kwargs["num_workers"] > 0 else False,
                **kwargs,
            )
        self.eval_index += 1

    @property
    def num_actors(self):
        return len(self.eval_actors)

    def _maybe_peval(self, strategy, counter, **kwargs):
        if self.eval_every > 0 and counter % self.eval_every == 0:
            self._peval(strategy, **kwargs)

    def after_training_iteration(self, strategy: "BaseSGDTemplate", **kwargs):
        """Periodic eval controlled by `self.eval_every`."""
        if self.peval_mode == "iteration":
            self._maybe_peval(strategy, strategy.clock.train_exp_iterations, **kwargs)

    def create_actors(self, actor_args, num_actors):
        actors = []
        # Make sure we use the "0" device for evaluation, actual devices will
        # be managed by ray
        if "device" in actor_args:
            device = actor_args["device"]
        else:
            device = "cuda"

        if device != "cpu":
            actor_args.update({"device": "cuda:0"})

        # Create actors for the strategy model
        for i in range(num_actors):
            actor = EvaluationActor.options(
                num_cpus=self.num_cpus, num_gpus=self.num_gpus
            ).remote(evaluator=self.evaluator, **actor_args)
            actors.append(actor)
        return actors
