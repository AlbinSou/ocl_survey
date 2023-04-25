import atexit
import copy
import os
from typing import Dict, List, Tuple

import torch
from torch.optim import SGD

import ray
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from toolkit.json_logger import JSONLogger


class BlockingScheduler:
    def __init__(self, max_launched):
        self.max_launched = max_launched
        self.task_queue = []
        self.running_tasks = []
        atexit.register(self.close)

    def schedule(self, function, *args, **kwargs):
        task = function.remote(*args)
        if len(self.task_queue) + len(self.running_tasks) > self.max_launched:
            self.task_queue.append(task)
            self._wait_for_task_to_complete()
        else:
            self.running_tasks.append(task)
        return task

    def _wait_for_task_to_complete(self):
        while True:
            ready, _ = ray.wait(self.running_tasks, num_returns=1)
            self.running_tasks.remove(ready[0])
            if not self.task_queue:
                break
            next_task = self.task_queue.pop(0)
            if len(self.task_queue) + \
                    len(self.running_tasks) > self.max_launched:
                self.task_queue.insert(0, next_task)
            else:
                self.running_tasks.append(next_task)
                break

    def close(self):
        ray.get(self.task_queue + self.running_tasks)


# change this if you want more/less resource per worker
@ray.remote(num_cpus=2, num_gpus=0.1)
class EvaluationActor(object):
    """Parallel Evaluation Actor.

    Methods called on different actors can execute in parallel, and methods
    called on the same actor are executed serially in the order that they
    are called. Methods on the same actor will share state with one another,
    as shown below."""

    def __init__(self, logdir="actor_log_dir", metrics=[], **strat_args):
        """Constructor.

        Remember to pass an evaluator to the model. Use different logdir for each actor.

        :param strat_args:
        """

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        filename_json = os.path.join(logdir, "logs.json")
        fp = open(filename_json, "w")
        fp.close()

        self.json_logger = JSONLogger(filename_json)

        evaluator = EvaluationPlugin(
            *metrics,
            loggers=[
                TensorboardLogger(tb_log_dir=logdir),
                self.json_logger,
            ],
        )

        peval_args = {"evaluator": evaluator}

        # NOTE: we need a stateful actor to keep the same
        # logger for each evaluation
        # step and to serialize the eval calls.
        self.strat = Naive(
            model=None,
            optimizer=None,
            **peval_args,
            **strat_args,
        )

    def eval(self, model, clock, stream, **kwargs):
        self.strat.model = model
        self.strat.clock = clock
        self.strat.eval(stream, **kwargs)

    def write_files(self):
        self.json_logger.update_json()


class ParallelEval(SupervisedPlugin):
    """Schedules periodic evaluation during training."""

    def __init__(
        self,
        metrics,
        results_dir,
        eval_every=-1,
        do_initial=False,
        num_actors=1,
        mode="iteration",
        max_launched=100,
        **actor_args
    ):
        """Init.

        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        :param do_initial: whether to evaluate before each `train` call.
            Occasionally needed becuase some metrics need to know the
            accuracy before training.
        """
        super().__init__()
        self.metrics = metrics
        self.mode = mode
        self.results_dir = results_dir
        self.num_actors = num_actors
        self.eval_actors = self.create_actors(actor_args, num_actors)
        self.eval_every = eval_every
        self.do_initial = do_initial and eval_every > -1

        self.scheduler = BlockingScheduler(max_launched=max_launched)

    def before_training(self, strategy, **kwargs):
        """Eval before each learning experience.

        Occasionally needed because some metrics need the accuracy before
        training.
        """
        if self.do_initial:
            self._peval(strategy, **kwargs)

    def after_training_exp(self, strategy, **kwargs):
        """Final eval after a learning experience."""
        self._peval(strategy, **kwargs)

    def before_training_exp(self, strategy, **kwargs):
        # Update stream refs
        self.stream_refs = []
        for el in strategy._eval_streams:
            self.stream_refs.append(ray.put(el))

    def _peval(self, strategy, **kwargs):
        # you can run multiple actors in parallel here.
        clock_ref = ray.put(strategy.clock)
        for stream_ref in self.stream_refs:

            # Strategy model
            actor = self.eval_actors[strategy.clock.train_iterations %
                                     self.num_actors]

            self.scheduler.schedule(
                actor.eval,
                copy.deepcopy(strategy.model).cpu(),
                clock_ref,
                stream_ref,
                persistent_workers=True if kwargs["num_workers"] > 0 else False,
                **kwargs,
            )

    def _maybe_peval(self, strategy, counter, **kwargs):
        if self.eval_every > 0 and counter % self.eval_every == 0:
            # print("LAUNCHING EVAL ON ITERATION {}".format(counter))
            self._peval(strategy, **kwargs)

    def after_training_iteration(self, strategy: "BaseSGDTemplate", **kwargs):
        """Periodic eval controlled by `self.eval_every`."""
        if self.mode == "iteration":
            self._maybe_peval(
                strategy,
                strategy.clock.train_exp_iterations,
                **kwargs)

    def create_actors(self, actor_args, num_actors):
        actors = []
        # Make sure we use the "0" device for evaluation, actual devices will
        # be managed by ray
        actor_args.update({"device": "cuda:0"})
        # Create actors for the strategy model
        for i in range(num_actors):
            actor = EvaluationActor.remote(
                self.results_dir, self.metrics, **actor_args)
            actors.append(actor)
        return actors


if __name__ == "__main__":
    ray.init(num_cpus=12, num_gpus=2)

    from avalanche.benchmarks import SplitMNIST

    scenario = SplitMNIST(5)
    model = SimpleMLP(10)
    optimizer = SGD(model.parameters(), lr=0.01)

    # NOTE: eval_every must be -1 to disable the main strategy's PeriodicEval
    # because we are going to use ParallelEval to evaluate the model.
    strat = Naive(
        model=model,
        optimizer=optimizer,
        train_mb_size=128,
        train_epochs=1,
        eval_every=-1,
        eval_mb_size=512,
        device="cuda",
        plugins=[
            ParallelEval(
                metrics=accuracy_metrics(stream=True),
                results_dir="./results",
                eval_every=1,
                num_actors=3,
            )
        ],
    )

    for t, experience in enumerate(scenario.train_stream):
        strat.train(
            experience, eval_streams=[
                scenario.test_stream[: t + 1]], num_workers=0
        )
