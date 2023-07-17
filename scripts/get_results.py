#!/usr/bin/env python3
import argparse

import src.toolkit.post_metrics as metrics_utils


def main(args):
    frames = metrics_utils.extract_results(args.results_dir)

    df_training = frames["training"]

    if "continual" in frames:
        df_continual = frames["continual"]
    else:
        df_continual = None

    if "probing" in frames:
        df_probing = frames["probing"]
    else:
        df_probing = None


    # Final average accuracy

    mean, std = metrics_utils.compute_mean_std_metric(
        df_training, "Top1_Acc_Stream/eval_phase/test_stream/Task000"
    )

    print()
    print(f"Final Average Accuracy mean: {mean}")
    print(f"Final Average Accuracy std: {std}")

    if df_continual is not None:
        #try:
        df_continual = metrics_utils.compute_AAA(
            df_continual, base_name="Top1_Acc_Stream/eval_phase/valid_stream/Task000"
        )
        mean, std = metrics_utils.compute_mean_std_metric(df_continual, "AAA")
        print()
        print(f"Final AAA mean: {mean}")
        print(f"Final AAA std: {std}")
        df_continual = metrics_utils.compute_wcacc(df_continual, num_tasks=20)
        mean, std = metrics_utils.compute_mean_std_metric(
            df_continual, "WCAcc"
        )
        print()
        print(f"Final WCAcc mean: {mean}")
        print(f"Final WCAcc std: {std}")
        #except Exception:
        #    pass

    if df_probing is not None:
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
