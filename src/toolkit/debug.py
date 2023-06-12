#!/usr/bin/env python3
import pdb
from typing import Callable, Union

from avalanche.training.plugins import SupervisedPlugin

def set_trace(strategy, **kwargs):
    pdb.set_trace()

def template_function(condition: Union[bool, Callable]) -> Callable:
    if isinstance(condition, bool):
        if condition: 
            return set_trace
        else:
            return lambda *args, **kwargs: None

    if isinstance(condition, Callable):
        def set_cond_trace(strategy, **kwargs):
            if condition(strategy): 
                pdb.set_trace()
        return set_cond_trace


class PdbPlugin(SupervisedPlugin):
    """ Debugging plugin that goes into debugger at given steps """
    def __init__(self, before_training=False,
        before_training_exp=False, after_eval_iteration=False, 
        before_train_dataset_adaptation=False,
        after_train_dataset_adaptation=False, before_training_epoch=False,
        before_training_iteration=False, before_forward=False,
        after_forward=False, before_backward=False, after_backward=False,
        after_training_iteration=False, before_update=False,
        after_update=False, after_training_epoch=False,
        after_training_exp=False, after_training=False,
        before_eval=False, before_eval_dataset_adaptation=False,
        after_eval_dataset_adaptation=False, before_eval_exp=False,
        after_eval_exp=False, after_eval=False, before_eval_iteration=False,
        before_eval_forward=False, after_eval_forward=False):
    

        super().__init__()

        self.priority = 2

        self.before_training = template_function(before_training)
        self.before_training_exp = template_function(before_training_exp)
        self.before_train_dataset_adaptation = template_function(before_train_dataset_adaptation)
        self.after_eval_iteration = template_function(after_eval_iteration)
        self.after_train_dataset_adaptation = template_function(after_train_dataset_adaptation)
        self.before_training_epoch = template_function(before_training_epoch)
        self.before_training_iteration = template_function(before_training_iteration)
        self.before_forward = template_function(before_forward)
        self.after_forward = template_function(after_forward)
        self.before_backward = template_function(before_backward)
        self.after_backward = template_function(after_backward)
        self.after_training_iteration = template_function(after_training_iteration)
        self.before_update = template_function(before_update)
        self.after_update = template_function(after_update)
        self.after_training_epoch = template_function(after_training_epoch)
        self.after_training_exp = template_function(after_training_exp)
        self.after_training = template_function(after_training)
        self.before_eval = template_function(before_eval)
        self.before_eval_dataset_adaptation = template_function(before_eval_dataset_adaptation)
        self.after_eval_dataset_adaptation = template_function(after_eval_dataset_adaptation)
        self.before_eval_exp = template_function(before_eval_exp)
        self.after_eval_exp = template_function(after_eval_exp)
        self.after_eval = template_function(after_eval)
        self.before_eval_iteration = template_function(before_eval_iteration)
        self.before_eval_forward = template_function(before_eval_forward)
        self.after_eval_forward = template_function(after_eval_forward)
