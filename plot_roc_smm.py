"""
plot_roc_smm.py - ROC curves for fake vs true SMM cross-validation predictions.

Requires per-peptide predictions produced by the *patched* run_cv.py
(see run_cv ROC patch). Expected columns:
    allele, method (fake|true), y_true, y_pred

A peptide is a "binder" if y_true >= threshold. The default threshold 0.4256
is the standard 500 nM affinity cutoff on the 1-log50k scale:
    1 - log10(500) / log10(50000) = 0.4256
Adjust with --threshold if your targets use a different transform.

    python plot_roc_smm.py --preds cv_predictions.csv --out smm_roc.png
    python plot_roc_smm.py --preds cv_predictions.csv --allele A0201 --out roc_A0201.png

NOTE on pooling: pooling peptides across alleles mixes different score
distributions, so a single pooled ROC is only a rough summary. For a clean
comparison use --allele, or read the per-allele AUC table this script prints.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

DEFAULT_THRESHOLD = 1 - np.log10(500) / np.log10(50000)  # 0.4256


def roc_for(df, threshold):
    """Return (fpr, tpr, auc) or None if only one class is present."""
    y = (df.y_true.values >= threshold).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    fpr, tpr, _ = roc_curve(y, df.y_pred.values)
    return fpr, tpr, auc(fpr, tpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="cv_predictions.csv")
    ap.add_argument("--allele", default=None,
                    help="restrict to one allele (recommended). Omit = pool all.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="binder cutoff on the y_true scale (default 0.4256 = 500 nM)")
    ap.add_argument("--out", default="smm_roc.png")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    if args.allele:
        df = df[df.allele == args.allele]
        scope = f"allele {args.allele}"
    else:
        scope = f"all {df.allele.nunique()} alleles pooled"

    # ---- per-allele AUC table (the rigorous comparison) ----
    print(f"{'allele':<10}{'N':>7}{'AUC_fake':>10}{'AUC_true':>10}{'dAUC':>9}")
    print("-" * 46)
    rows = []
    for allele, sub in df.groupby("allele"):
        rf = roc_for(sub[sub.method == "fake"], args.threshold)
        rt = roc_for(sub[sub.method == "true"], args.threshold)
        if rf is None or rt is None:
            continue
        n = (sub.method == "true").sum()
        rows.append((allele, n, rf[2], rt[2]))
        print(f"{allele:<10}{n:>7}{rf[2]:>10.4f}{rt[2]:>10.4f}{rf[2]-rt[2]:>9.4f}")
    if rows:
        af = np.mean([r[2] for r in rows]); at = np.mean([r[3] for r in rows])
        print("-" * 46)
        print(f"{'MEAN':<10}{'':>7}{af:>10.4f}{at:>10.4f}{af-at:>9.4f}")

    # ---- pooled (or single-allele) ROC figure ----
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for method, color in [("fake", "#d9534f"), ("true", "#0275d8")]:
        r = roc_for(df[df.method == method], args.threshold)
        if r is None:
            continue
        fpr, tpr, a = r
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{method}  (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(xlim=(0, 1), ylim=(0, 1),
           xlabel="false positive rate", ylabel="true positive rate",
           title=f"ROC: fake vs true CV predictions\n({scope}, binder >= {args.threshold:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()