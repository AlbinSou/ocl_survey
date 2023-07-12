#!/usr/bin/env python3
import argparse

import src.toolkit.post_metrics as metrics_utils


def main(args):
    frames = metrics_utils.extract_results(args.results_dir)

    df_training = frames["training"]
    df_continual = frames["continual"]
    df_probing = frames["probing"]

    # Final average accuracy

    mean, std = metrics_utils.compute_mean_std_metric(
        df_training, "Top1_Acc_Stream/eval_phase/test_stream/Task000"
    )

    print()
    print(f"Final Average Accuracy mean: {mean}")
    print(f"Final Average Accuracy std: {std}")

    df_continual = metrics_utils.compute_AAA(
        df_continual, base_name="Top1_Acc_Stream/eval_phase/valid_stream/Task000"
    )

    mean, std = metrics_utils.compute_mean_std_metric(df_continual, "AAA")

    print()
    print(f"Final Average Accuracy mean: {mean}")
    print(f"Final Average Accuracy std: {std}")

    mean, std = metrics_utils.compute_mean_std_metric(
        df_probing, "Top1_Acc_Stream/eval_phase/test_stream/Task000"
    )

    print()
    print(f"Final PROBED REPRESENTATION Average Accuracy mean: {mean}")
    print(f"Final PROBED REPRESENTATION Average Accuracy std: {std}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()
    main(args)
