import os
import sys
import subprocess
from argparse import ArgumentParser

import pandas as pd


def run_command(cmd):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = ArgumentParser(description="Run non-nested CV for one or more alleles")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="Path to data directory containing allele folders")
    parser.add_argument("--results_dir", required=True, type=str,
                        help="Path to output results directory")
    parser.add_argument("--blosum_file", required=True, type=str,
                        help="Path to BLOSUM file")
    parser.add_argument("--alleles", nargs="+", required=True,
                        help="Alleles to run, e.g. A0101 A0201")
    parser.add_argument("--folds", nargs="+",
                        default=["000", "001", "002", "003", "004"],
                        help="Fold IDs to run")
    parser.add_argument("--hidden_size", required=True, type=int,
                        help="Hidden layer size")
    parser.add_argument("--learning_rate", required=True, type=float,
                        help="Learning rate")
    parser.add_argument("--epochs", required=True, type=int,
                        help="Number of epochs")
    parser.add_argument("--binder_threshold", default=None, type=float,
                        help="Optional threshold for AUC / binder count")
    args = parser.parse_args()

    print("Parsed arguments:", args, flush=True)
    print("Starting non-nested CV runner...", flush=True)

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    if not os.path.exists(args.blosum_file):
        raise FileNotFoundError(f"BLOSUM file not found: {args.blosum_file}")

    os.makedirs(args.results_dir, exist_ok=True)

    all_metrics = []
    python_exe = sys.executable

    for allele in args.alleles:
        print(f"\nProcessing allele: {allele}", flush=True)

        allele_data_dir = os.path.join(args.data_dir, allele)
        allele_results_dir = os.path.join(args.results_dir, allele)

        if not os.path.exists(allele_data_dir):
            raise FileNotFoundError(f"Allele data directory not found: {allele_data_dir}")

        os.makedirs(allele_results_dir, exist_ok=True)
        allele_metrics = []

        for fold in args.folds:
            print(f"Processing fold: {fold}", flush=True)

            train_file = os.path.join(allele_data_dir, f"c{fold}")
            infer_file = os.path.join(allele_data_dir, f"f{fold}")
            fold_name = f"fold_{fold}"
            fold_results_dir = os.path.join(allele_results_dir, fold_name)
            os.makedirs(fold_results_dir, exist_ok=True)

            if not os.path.exists(train_file):
                raise FileNotFoundError(f"Missing train file: {train_file}")
            if not os.path.exists(infer_file):
                raise FileNotFoundError(f"Missing inference file: {infer_file}")

            model_path = os.path.join(fold_results_dir, "nnalign_model.pt")
            predictions_csv = os.path.join(fold_results_dir, "predictions.csv")
            metrics_csv = os.path.join(fold_results_dir, "metrics.csv")

            train_cmd = [
                python_exe, "train_nnalign.py",
                "--train_data", train_file,
                "--valid_data", infer_file,
                "--hidden_size", str(args.hidden_size),
                "--learning_rate", str(args.learning_rate),
                "--epochs", str(args.epochs),
                "--output_dir", fold_results_dir,
                "--blosum_file", args.blosum_file
            ]
            run_command(train_cmd)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Expected model file was not created: {model_path}")

            infer_cmd = [
                python_exe, "infer_nnalign.py",
                "--inference_data", infer_file,
                "--parameter_file", model_path,
                "--output_dir", fold_results_dir,
                "--blosum_file", args.blosum_file
            ]
            run_command(infer_cmd)

            if not os.path.exists(predictions_csv):
                raise FileNotFoundError(f"Expected predictions file was not created: {predictions_csv}")

            eval_cmd = [
                python_exe, "eval_predictions.py",
                "--predictions_csv", predictions_csv,
                "--allele", allele,
                "--fold", fold_name,
                "--cv_type", "non_nested",
                "--output_csv", metrics_csv
            ]

            if args.binder_threshold is not None:
                eval_cmd.extend(["--binder_threshold", str(args.binder_threshold)])

            run_command(eval_cmd)

            if not os.path.exists(metrics_csv):
                raise FileNotFoundError(f"Expected metrics file was not created: {metrics_csv}")

            fold_metrics = pd.read_csv(metrics_csv)
            allele_metrics.append(fold_metrics)
            all_metrics.append(fold_metrics)

        if allele_metrics:
            allele_df = pd.concat(allele_metrics, ignore_index=True)
            allele_summary_path = os.path.join(allele_results_dir, "cv_results.csv")
            allele_df.to_csv(allele_summary_path, index=False)
            print(f"Saved allele summary to {allele_summary_path}", flush=True)

    if all_metrics:
        final_df = pd.concat(all_metrics, ignore_index=True)
        final_summary_path = os.path.join(args.results_dir, "all_alleles_cv_results.csv")
        final_df.to_csv(final_summary_path, index=False)
        print(f"Saved overall summary to {final_summary_path}", flush=True)


if __name__ == "__main__":
    main()