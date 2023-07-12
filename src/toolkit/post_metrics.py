#!/usr/bin/env python3
"""
Metrics computed from the logs

- Forgetting
- CumulativeForgetting
- WC_Acc
- Min_Acc

"""

import numpy as np
import pandas as pd

from src.toolkit.process_results import extract_results


def compute_forgetting(dataframe, metric_name, prefix="Forgetting_"):
    """
    Computes forgetting for a given metric name (difference from maximum value)

    Adds it as a column
    """

    df = dataframe.sort_values("mb_index")

    forg_metric_name = prefix + metric_name

    df[forg_metric_name] = df.groupby("seed")[metric_name].cummax() - df[metric_name]

    return df


def compute_average(dataframe, metric_names, avg_metric_name):
    """
    Computes average of several metric names given

    Adds it as a column
    """

    df = dataframe.sort_values("mb_index")
    # df = df.dropna(subset=metric_names, how="all")

    df[avg_metric_name] = df[metric_names].mean(axis=1)

    return df


def compute_average_forgetting(
    dataframe,
    num_exp,
    base_name="Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp",
    name="Average_Forgetting",
):
    """
    Computes average forgetting and adds it as a dataframe columns
    """

    prefix = "Forgetting_"
    metric_names = []
    for i in range(num_exp):
        mname = base_name + f"{i:03d}"
        dataframe = compute_forgetting(dataframe, mname)
        metric_names.append(prefix + mname)

    df = compute_average(dataframe, metric_names, name)
    return df


def annotate_splits(dataframe, splits):
    for task, (sp1, sp2) in splits.items():
        df.loc[(df.mb_index > sp1) & (df.mb_index <= sp2), "training_task"] = task
    return df


def min_acc(dataframe, splits):
    df = annotate_splits(dataframe, splits)
    for task in range(len(splits)):
        metric_name = f"Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task:03d}"
        df["Min_" + metric_name] = df.groupby("seed")[metric_name].cummin()


def compute_AAA(
    dataframe,
    base_name="Top1_Acc_Stream/eval_phase/valid_stream/Task000",
):
    """
    Computes average anytime accuracy
    """
    df = dataframe.sort_values(["seed", "mb_index"])
    df["cumulative_sum"] = df.groupby("seed")[base_name].cumsum()
    df["count"] = df.groupby("seed").cumcount() + 1
    df["AAA"] = df["cumulative_sum"] / df["count"]
    return df


def compute_mean_std_metric(dataframe, metric_name, final_step=True):
    df = dataframe
    max_index = df["mb_index"].max()
    mean = df.loc[df["mb_index"] == max_index, metric_name].mean()
    std = df.loc[df["mb_index"] == max_index, metric_name].std()
    return mean, std


if __name__ == "__main__":
    results_dir = "/DATA/ocl_survey/er_split_cifar100_20_2000/"

    frames = extract_results(results_dir)
    df = frames["training"]

    df = compute_average_forgetting(df, 20)
