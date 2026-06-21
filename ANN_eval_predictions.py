import os
from argparse import ArgumentParser

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, roc_auc_score


def compute_metrics(df, binder_threshold=None):
    if "target" not in df.columns or "prediction" not in df.columns:
        raise ValueError("Input CSV must contain 'target' and 'prediction' columns.")

    y_true = df["target"].to_numpy()
    y_pred = df["prediction"].to_numpy()

    metrics = {}

    pcc, _ = pearsonr(y_true, y_pred)
    metrics["pcc"] = float(pcc)

    mse = mean_squared_error(y_true, y_pred)
    metrics["mse"] = float(mse)

    if binder_threshold is not None:
        y_bin = (y_true >= binder_threshold).astype(int)

        if len(set(y_bin)) < 2:
            metrics["auc"] = None
        else:
            metrics["auc"] = float(roc_auc_score(y_bin, y_pred))

    return metrics


def main():
    parser = ArgumentParser(description="Evaluate predictions from NNAlign-like model")
    parser.add_argument("--predictions_csv", required=True, type=str,
                        help="Path to predictions.csv")
    parser.add_argument("--allele", default=None, type=str,
                        help="Allele name, e.g. A0201")
    parser.add_argument("--fold", default=None, type=str,
                        help="Fold name, e.g. fold_000")
    parser.add_argument("--cv_type", default="non_nested", type=str,
                        help="CV type label, e.g. non_nested or nested")
    parser.add_argument("--binder_threshold", default=None, type=float,
                        help="Optional threshold to binarize target for AUC")
    parser.add_argument("--output_csv", default=None, type=str,
                        help="Optional path to save one-row metrics CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    metrics = compute_metrics(df, binder_threshold=args.binder_threshold)

    allele = args.allele if args.allele is not None else "NA"
    fold = args.fold if args.fold is not None else "NA"

    print(f"Allele: {allele}")
    print(f"Fold: {fold}")
    print(f"N peptides: {len(df)}")
    print(f"PCC: {metrics['pcc']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")

    if "auc" in metrics:
        if metrics["auc"] is None:
            print("AUC: undefined (only one class after thresholding)")
        else:
            print(f"AUC: {metrics['auc']:.6f}")

    if args.output_csv is not None:
        n_binders = None
        if args.binder_threshold is not None:
            n_binders = int((df["target"].to_numpy() >= args.binder_threshold).sum())

        out_df = pd.DataFrame([{
            "allele": allele,
            "fold": fold,
            "cv_type": args.cv_type,
            "n_peptides": len(df),
            "n_binders": n_binders,
            "pcc": metrics["pcc"],
            "mse": metrics["mse"],
            "auc": metrics.get("auc", None),
            "binder_threshold": args.binder_threshold
        }])

        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()