#!/usr/bin/env python3
import os
from typing import TYPE_CHECKING, List
import jsonlines
import atexit

import torch
from avalanche.logging import BaseLogger
import collections


class JSONLogger(BaseLogger):
    """ Logs metric dict into a jsonlines file 
    where one step of evaluation is logged as a
    separate line. The logger empties the 
    current metric dict to the end of the file when 
    update_json is called. This allows for multiple 
    writings to be done in parallel without compromising
    the output file """

    def __init__(self, filename, autoupdate=False):
        super().__init__()
        self.filename = filename
        self.metric_dict = collections.defaultdict(lambda: {})
        self.autoupdate = autoupdate
        
        # Always write file at exit
        atexit.register(self.update_json)

    def log_single_metric(self, name, value, x_plot):
        self.metric_dict[x_plot][name] = value
        if self.autoupdate:
            self.update_json()

    def _convert_to_records(self, metric_dict):
        records = []
        for step, mdict in metric_dict.items():
            new_dict = {"step": step}
            new_dict.update(mdict)
            records.append(new_dict)
        return records

    def update_json(self):
        # Reset metric dict and put info in file
        records = self._convert_to_records(self.metric_dict)
        with jsonlines.open(self.filename, mode="a") as writer:
            writer.write_all(records)
        self.metric_dict = collections.defaultdict(lambda: {})
