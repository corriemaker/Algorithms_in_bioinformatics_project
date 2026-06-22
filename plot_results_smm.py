"""
plot_results.py - Figures for the SMM fake-vs-true cross-validation results.

    python plot_results.py --summary cv_summary.csv --fake_grid cv_fake_grid.csv \
        --eps 0.01 --out smm_cv_figures.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="cv_summary.csv")
    ap.add_argument("--fake_grid", default="cv_fake_grid.csv")
    ap.add_argument("--eps", type=float, default=0.01,
                    help="epsilon slice to use for the complexity curve")
    ap.add_argument("--out", default="smm_cv_figures.png")
    args = ap.parse_args()

    s = pd.read_csv(args.summary)
    g = pd.read_csv(args.fake_grid)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # ---- (A) fake-best vs true, per allele ----
    lo = min(s.true_pcc.min(), s.fake_best_pcc.min()) - 0.03
    hi = max(s.true_pcc.max(), s.fake_best_pcc.max()) + 0.03
    sc = ax[0].scatter(s.true_pcc, s.fake_best_pcc, c=np.log10(s.n_peptides),
                       cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="fake = true")
    ax[0].set(xlim=(lo, hi), ylim=(lo, hi),
              xlabel="true (nested) PCC", ylabel="fake (best-over-grid) PCC",
              title="Fake vs true, per allele")
    ax[0].legend(loc="upper left", fontsize=8)
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("log10(N peptides)", fontsize=8)

    # ---- (B) optimism (gap) vs data-set size ----
    ax[1].axhline(0, color="grey", lw=1)
    ax[1].scatter(s.n_peptides, s.gap, s=40, edgecolor="k", linewidth=0.3, color="#d9534f")
    # log-linear trend
    x = np.log10(s.n_peptides); y = s.gap.values
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax[1].plot(10**xs, a + b*xs, "b-", lw=1.5,
               label=f"trend (r={np.corrcoef(x, y)[0,1]:+.2f})")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="N peptides (allele)", ylabel="optimism  (fake - true) PCC",
              title="Optimism shrinks with data")
    ax[1].legend(loc="upper right", fontsize=8)
    # annotate the worst offender
    am = s.loc[s.gap.idxmax()]
    ax[1].annotate(am.allele, (am.n_peptides, am.gap),
                   textcoords="offset points", xytext=(6, -2), fontsize=8)

    # ---- (C) complexity curve: fake PCC vs lambda ----
    ge = g[np.isclose(g.epsilon, args.eps)]
    for allele, sub in ge.groupby("allele"):
        sub = sub.sort_values("lambda")
        ax[2].plot(sub["lambda"], sub.fake_pcc, color="grey", alpha=0.18, lw=0.8)
    mean_curve = ge.groupby("lambda").fake_pcc.mean()
    ax[2].plot(mean_curve.index, mean_curve.values, "o-", color="#0275d8",
               lw=2.2, ms=5, label="mean over alleles")
    peak = mean_curve.idxmax()
    ax[2].axvline(peak, color="green", ls=":", lw=1.2, label=f"peak lambda={peak:g}")
    ax[2].set_xscale("log")
    ax[2].set(xlabel="lambda  (low = complex/over-fit, high = simple/under-fit)",
              ylabel=f"fake 5-fold PCC  (eps={args.eps:g})",
              title="SMM complexity curve")
    ax[2].legend(loc="lower center", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()