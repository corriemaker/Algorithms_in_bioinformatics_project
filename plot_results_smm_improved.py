"""
plot_results_smm.py - Figures for the SMM fake-vs-true cross-validation results.

Main figure (2x3):
    (A) fake-best vs true PCC, per allele          [original]
    (B) optimism gap vs dataset size               [original]
    (C) SMM complexity curve (fake PCC vs lambda)   [original]
    (D) fake & true PCC vs N, paired per allele    [NEW: levels + convergence]
    (E) optimism gap vs model-selection instability [NEW: the mechanism]
    (F) distribution of the optimism gap            [NEW: one-sidedness]

Second figure:
    lambda-selection stability heatmap (how often each outer fold
    picked each lambda, per allele)                 [NEW]

PCC = Pearson Correlation Coefficient (linear correlation between predicted
and measured 1-log50k binding values; +1 = perfect, 0 = none).

    python plot_results_smm.py --summary cv_summary.csv --fake_grid cv_fake_grid.csv \
        --eps 0.01 --out smm_cv_figures.png --out_heatmap smm_lambda_stability.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ---- helpers --------------------------------------------------------------

def parse_selected(col):
    """';'-joined floats -> list[float]. Tolerates blanks/NaN."""
    if not isinstance(col, str) or col.strip() == "":
        return []
    return [float(x) for x in col.split(";") if x.strip() != ""]


def instability(lambdas):
    """Spread of the per-fold selected lambda, in log10 space.
    0 = every outer fold chose the same lambda (perfectly stable)."""
    if len(lambdas) < 2:
        return 0.0
    return float(np.std(np.log10(lambdas)))


# ---- main figure ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="cv_summary.csv")
    ap.add_argument("--fake_grid", default="cv_fake_grid.csv")
    ap.add_argument("--eps", type=float, default=0.01,
                    help="epsilon slice to use for the complexity curve")
    ap.add_argument("--out", default="smm_cv_figures.png")
    ap.add_argument("--out_heatmap", default="smm_lambda_stability.png")
    args = ap.parse_args()

    s = pd.read_csv(args.summary)
    g = pd.read_csv(args.fake_grid)

    # per-allele selected-lambda lists + instability score
    s = s.copy()
    s["sel_lambdas"] = s.selected_lambdas.apply(parse_selected)
    s["instability"] = s.sel_lambdas.apply(instability)
    s["n_distinct"] = s.sel_lambdas.apply(lambda L: len(set(L)))

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
    ax = axes.ravel()

    # ============ (A) fake-best vs true, per allele ============
    lo = min(s.true_pcc.min(), s.fake_best_pcc.min()) - 0.03
    hi = max(s.true_pcc.max(), s.fake_best_pcc.max()) + 0.03
    sc = ax[0].scatter(s.true_pcc, s.fake_best_pcc, c=np.log10(s.n_peptides),
                       cmap="viridis", s=42, edgecolor="k", linewidth=0.3)
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="fake = true")
    ax[0].set(xlim=(lo, hi), ylim=(lo, hi),
              xlabel="true (nested) PCC", ylabel="fake (best-over-grid) PCC",
              title="(A) Fake vs true, per allele")
    ax[0].legend(loc="upper left", fontsize=8)
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("log10(N peptides)", fontsize=8)

    # ============ (B) optimism (gap) vs data-set size ============
    ax[1].axhline(0, color="grey", lw=1)
    ax[1].scatter(s.n_peptides, s.gap, s=42, edgecolor="k", linewidth=0.3,
                  color="#d9534f")
    x = np.log10(s.n_peptides); y = s.gap.values
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax[1].plot(10**xs, a + b * xs, "b-", lw=1.5,
               label=f"trend (r={np.corrcoef(x, y)[0,1]:+.2f})")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="N peptides (allele)", ylabel="optimism  (fake - true) PCC",
              title="(B) Optimism shrinks with data")
    ax[1].legend(loc="upper right", fontsize=8)
    am = s.loc[s.gap.idxmax()]
    ax[1].annotate(am.allele, (am.n_peptides, am.gap),
                   textcoords="offset points", xytext=(6, -2), fontsize=8)

    # ============ (C) complexity curve: fake PCC vs lambda ============
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
              title="(C) SMM complexity curve")
    ax[2].legend(loc="lower center", fontsize=8)

    # ============ (D) fake & true PCC vs N, paired per allele ============
    # vertical segment per allele connects its true (lower) to fake (upper)
    order = s.sort_values("n_peptides")
    for _, r in order.iterrows():
        ax[3].plot([r.n_peptides, r.n_peptides], [r.true_pcc, r.fake_best_pcc],
                   color="0.7", lw=0.8, zorder=1)
    ax[3].scatter(order.n_peptides, order.fake_best_pcc, s=34, color="#d9534f",
                  edgecolor="k", linewidth=0.3, label="fake (best-over-grid)", zorder=3)
    ax[3].scatter(order.n_peptides, order.true_pcc, s=34, color="#0275d8",
                  edgecolor="k", linewidth=0.3, label="true (nested)", zorder=3)
    # log-linear trends for each
    xl = np.log10(order.n_peptides.values)
    for yv, col in [(order.fake_best_pcc.values, "#d9534f"),
                    (order.true_pcc.values, "#0275d8")]:
        bb, aa = np.polyfit(xl, yv, 1)
        xx = np.linspace(xl.min(), xl.max(), 50)
        ax[3].plot(10**xx, aa + bb * xx, color=col, lw=1.3, alpha=0.8)
    ax[3].set_xscale("log")
    ax[3].set(xlabel="N peptides (allele)", ylabel="PCC",
              title="(D) Both estimates rise — and converge — with N")
    ax[3].legend(loc="lower right", fontsize=8)

    # ============ (E) gap vs model-selection instability ============
    # jitter-free: instability is already continuous (std of log10 selected lambda)
    sc2 = ax[4].scatter(s.instability, s.gap, c=np.log10(s.n_peptides),
                        cmap="viridis", s=44, edgecolor="k", linewidth=0.3)
    if s.instability.std() > 0:
        bb, aa = np.polyfit(s.instability, s.gap, 1)
        xx = np.linspace(0, s.instability.max() * 1.02, 50)
        rr = np.corrcoef(s.instability, s.gap)[0, 1]
        ax[4].plot(xx, aa + bb * xx, "k-", lw=1.4, label=f"trend (r={rr:+.2f})")
        ax[4].legend(loc="upper left", fontsize=8)
    ax[4].axhline(0, color="grey", lw=0.8)
    ax[4].set(xlabel="selection instability  =  std( log10  selected-lambda )  over 5 outer folds",
              ylabel="optimism  (fake - true) PCC",
              title="(E) Unstable model-selection drives the optimism")
    cb2 = fig.colorbar(sc2, ax=ax[4]); cb2.set_label("log10(N peptides)", fontsize=8)

    # ============ (F) distribution of the optimism gap ============
    ax[5].hist(s.gap, bins=14, color="#5cb85c", edgecolor="k", linewidth=0.4)
    ax[5].axvline(0, color="grey", lw=1)
    mg = s.gap.mean()
    ax[5].axvline(mg, color="#d9534f", ls="--", lw=1.6,
                  label=f"mean = {mg:+.3f}")
    med = s.gap.median()
    ax[5].axvline(med, color="#0275d8", ls=":", lw=1.6,
                  label=f"median = {med:+.3f}")
    ax[5].set(xlabel="optimism  (fake - true) PCC", ylabel="number of alleles",
              title="(F) The gap is one-sided (fake never below true)")
    ax[5].legend(loc="upper right", fontsize=8)

    fig.suptitle("SMM fake (non-nested) vs true (nested) cross-validation  -  35 HLA-I alleles",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    # ============ second figure: lambda-selection stability heatmap ============
    grid_lambdas = sorted(g["lambda"].unique())
    lam_index = {round(l, 6): i for i, l in enumerate(grid_lambdas)}
    order = s.sort_values("n_peptides", ascending=True).reset_index(drop=True)
    M = np.zeros((len(order), len(grid_lambdas)), dtype=int)
    for i, r in order.iterrows():
        for lam in r.sel_lambdas:
            j = lam_index.get(round(lam, 6))
            if j is not None:
                M[i, j] += 1

    fh, axh = plt.subplots(figsize=(9.2, 11))
    im = axh.imshow(M, aspect="auto", cmap="magma_r", vmin=0, vmax=5)
    axh.set_xticks(range(len(grid_lambdas)))
    axh.set_xticklabels([f"{l:g}" for l in grid_lambdas], rotation=45, ha="right",
                        fontsize=8)
    axh.set_yticks(range(len(order)))
    axh.set_yticklabels([f"{r.allele}  (N={int(r.n_peptides)})" for _, r in order.iterrows()],
                        fontsize=6.5)
    axh.set_xlabel("lambda (complexity axis)  -  low = complex, high = simple")
    axh.set_ylabel("allele   (sorted small -> large N, top -> bottom)")
    axh.set_title("Lambda chosen by the 5 outer folds (nested CV)\n"
                  "one stable column = robust selection; spread = unstable", fontsize=11)
    # annotate counts >0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                axh.text(j, i, str(M[i, j]), ha="center", va="center",
                         fontsize=6, color="white" if M[i, j] >= 3 else "0.2")
    cbh = fh.colorbar(im, ax=axh, fraction=0.035, pad=0.02)
    cbh.set_label("# of outer folds (0-5) that selected this lambda")
    fh.tight_layout()
    fh.savefig(args.out_heatmap, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out_heatmap}")


if __name__ == "__main__":
    main()