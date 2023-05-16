#!/usr/bin/env python3
from typing import List, Union

import numpy as np


def linear_schedule(start_value, coefficient, min_value=None, max_value=None):
    """
    Returns a scheduling function that starts from start value
    and increases or decreases with a ramp of given coefficient
    """
    def _lambda(iter_count):
        _min_value = min_value
        _max_value = max_value
        value = start_value + coefficient*iter_count
        if min_value is None:
            _min_value = value
        if max_value is None:
            _max_value = value
        return min(max(_min_value, value), _max_value)

    return _lambda
