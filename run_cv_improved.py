"""
run_cv.py - Fake (non-nested) vs True (nested) cross-validation for SMM.

Each allele's five c-files are loaded and encoded once, 
then reused across every fold and every hyper-parameter setting. 
Hyper-parameters are selected per allele, meaning
each allele has its own optimal lambda.

Per allele (folds = c000..c004 used as 5 folds):

  FAKE  (single-layer 5-fold CV), for each HP:
      for each test fold t: train on the other 4 folds, predict t
      concatenate the 5 test-fold predictions -> one PCC/MSE
      "fake" estimate = BEST PCC over the grid  (HP chosen on the very folds
      used to report it so it's optimistically biased)

  TRUE  (nested 5-fold CV):
      for each OUTER (eval) fold t:
          inner 4-fold CV on the other 4 folds:
              for each HP:
                  for each inner validation fold v: train on the other 3, predict v
                  concatenate the 4 validation preds -> inner PCC
              pick HP* = best inner PCC
          retrain once on all 4 development folds with HP* and predict the outer fold
      concatenate the 5 outer-fold predictions -> true PCC/MSE
      the 5 selected HP* are reported (their consistency = robustness)

Performance is computed on CONCATENATED predictions, never as an average of
per-fold correlations.

Example
-------
First sanity run (one allele, one lambda):
    python run_cv.py --data_dir DATA --alleles A0101 --lambdas 0.01

Full study:
    python run_cv.py --data_dir DATA --alleles all \
        --lambdas 0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30 \
        --epsilons 0.01,0.05,0.1 --epochs 200 --jobs 8
"""

import argparse
import itertools
import os
import sys

import numpy as np

import smm_gd as smm



#Data handling

def discover_alleles(data_dir):
    alleles_dir = os.path.join(data_dir, "Alleles")          
    out = []
    for name in sorted(os.listdir(alleles_dir)):
        if os.path.isfile(os.path.join(alleles_dir, name, "c000")):
            out.append(name)
    return out


def load_allele(allele, data_dir, scheme, alphabet, n_folds=5):
    """Load c000..c004 for one allele as a list of (X, y) folds."""
    folds = []
    for n in range(n_folds):
        path = os.path.join(data_dir, "Alleles", allele, f"c{n:03d}")
        X, y, _ = smm.load_dataset(path, scheme, alphabet)
        folds.append((X, y))
    return folds


def _cat(folds, idxs):
    X = np.vstack([folds[i][0] for i in idxs])
    y = np.concatenate([folds[i][1] for i in idxs])
    return X, y



#Cross-validation

def fake_cv(folds, hp, seed, return_preds=False):
    """Single-layer 5-fold CV for one HP. Returns (pcc, mse)."""
    lamb, eps, epochs = hp
    n = len(folds)
    yt, yp = [], []
    for t in range(n):
        tr = [i for i in range(n) if i != t]
        Xtr, ytr = _cat(folds, tr)
        w = smm.train_smm(Xtr, ytr, lamb, eps, epochs, seed)
        Xte, yte = folds[t]
        yt.append(yte)
        yp.append(smm.predict(Xte, w))
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    if return_preds:
        return smm.pcc(yt, yp), smm.mse(yt, yp), yt, yp
    return smm.pcc(yt, yp), smm.mse(yt, yp)


def true_cv(folds, grid, seed):
    """Nested 5-fold CV. Returns (pcc, mse, selected_hp_per_outer_fold)."""
    n = len(folds)
    outer_y, outer_pred, selected = [], [], []

    for t in range(n):                                   #outer / eval fold
        dev = [i for i in range(n) if i != t]            #4 development folds
        best_idx, best_pcc = None, -np.inf

        for hi, (lamb, eps, epochs) in enumerate(grid):  #inner HP search
            vy, vp = [], []
            for v in dev:                                #inner validation fold
                tr = [i for i in dev if i != v]          #3 training folds
                Xtr, ytr = _cat(folds, tr)
                w = smm.train_smm(Xtr, ytr, lamb, eps, epochs, seed)
                Xv, yv = folds[v]
                vy.append(yv)
                vp.append(smm.predict(Xv, w))
            p = smm.pcc(np.concatenate(vy), np.concatenate(vp))
            if not np.isnan(p) and p > best_pcc:
                best_pcc, best_idx = p, hi

        if best_idx is None:
            raise ValueError("All inner-CV hyperparameter scores were NaN")

        # Retrain once on all 4 outer-training folds with the selected HP,
        # then evaluate on the untouched outer fold.
        lamb, eps, epochs = grid[best_idx]
        Xdev, ydev = _cat(folds, dev)
        w_final = smm.train_smm(Xdev, ydev, lamb, eps, epochs, seed)

        Xt, yt = folds[t]
        outer_y.append(yt)
        outer_pred.append(smm.predict(Xt, w_final))
        selected.append(grid[best_idx])

    yt = np.concatenate(outer_y)
    yp = np.concatenate(outer_pred)
    return smm.pcc(yt, yp), smm.mse(yt, yp), selected, yt, yp



#Per-allele runner

def run_allele_folds(allele, folds, grid, seed):
    n_pep = int(sum(len(y) for _, y in folds))

    fake_curve = [(hp, *fake_cv(folds, hp, seed)) for hp in grid]
    best = max(fake_curve,
               key=lambda r: (r[1] if not np.isnan(r[1]) else -np.inf))
    fake_best_hp, fake_best_pcc, fake_best_mse = best

    true_pcc, true_mse, selected, true_yt, true_yp = true_cv(folds, grid, seed)
    _, _, fake_yt, fake_yp = fake_cv(folds, fake_best_hp, seed, return_preds=True)

    return {
        "allele": allele, "n_peptides": n_pep,
        "fake_curve": fake_curve,
        "fake_best_hp": fake_best_hp,
        "fake_best_pcc": fake_best_pcc, "fake_best_mse": fake_best_mse,
        "true_pcc": true_pcc, "true_mse": true_mse,
        "true_yt": true_yt, "true_yp": true_yp,
        "fake_yt": fake_yt, "fake_yp": fake_yp,
        "selected": selected,
    }


def run_allele(allele, data_dir, scheme, alphabet, grid, seed):
    """Top-level entry (loads + computes) so it is picklable for multiprocessing."""
    folds = load_allele(allele, data_dir, scheme, alphabet)
    return run_allele_folds(allele, folds, grid, seed)



#Output

def write_outputs(results, grid, prefix):
    #1: full fake curve (for plotting fake PCC vs lambda)
    with open(f"{prefix}_fake_grid.csv", "w") as f:
        f.write("allele,n_peptides,lambda,epsilon,epochs,fake_pcc,fake_mse\n")
        for r in results:
            for (lamb, eps, ep), p, m in r["fake_curve"]:
                f.write(f"{r['allele']},{r['n_peptides']},{lamb},{eps},{ep},"
                        f"{p:.6f},{m:.6f}\n")

    #2: per-allele summary (fake-best vs true, gap, selected HPs)
    with open(f"{prefix}_summary.csv", "w") as f:
        f.write("allele,n_peptides,fake_best_pcc,fake_best_lambda,fake_best_epsilon,"
                "true_pcc,gap,true_mse,fake_best_mse,selected_lambdas,selected_epsilons\n")
        for r in results:
            fl, fe, _ = r["fake_best_hp"]
            gap = r["fake_best_pcc"] - r["true_pcc"]
            sl = ";".join(str(hp[0]) for hp in r["selected"])
            se = ";".join(str(hp[1]) for hp in r["selected"])
            f.write(f"{r['allele']},{r['n_peptides']},{r['fake_best_pcc']:.6f},{fl},{fe},"
                    f"{r['true_pcc']:.6f},{gap:.6f},{r['true_mse']:.6f},"
                    f"{r['fake_best_mse']:.6f},{sl},{se}\n")

def write_predictions(results, prefix):
    with open(f"{prefix}_predictions.csv", "w") as f:
        f.write("allele,method,y_true,y_pred\n")
        for r in results:
            for yt, yp in zip(r["fake_yt"], r["fake_yp"]):
                f.write(f"{r['allele']},fake,{yt:.6f},{yp:.6f}\n")
            for yt, yp in zip(r["true_yt"], r["true_yp"]):
                f.write(f"{r['allele']},true,{yt:.6f},{yp:.6f}\n")

def print_summary(results):
    print(f"\n{'allele':<10}{'N':>7}{'fake_best':>11}{'true':>9}{'gap':>9}   selected lambdas")
    print("-" * 78)
    gaps = []
    for r in sorted(results, key=lambda x: x["n_peptides"]):
        gap = r["fake_best_pcc"] - r["true_pcc"]
        gaps.append(gap)
        sl = ",".join(str(hp[0]) for hp in r["selected"])
        print(f"{r['allele']:<10}{r['n_peptides']:>7}{r['fake_best_pcc']:>11.4f}"
              f"{r['true_pcc']:>9.4f}{gap:>9.4f}   [{sl}]")
    if gaps:
        print("-" * 78)
        print(f"mean optimism (fake_best - true) over {len(gaps)} alleles: "
              f"{np.mean(gaps):+.4f}")



#CLI

def _floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description="SMM fake-vs-true cross-validation")
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing one sub-folder per allele (each with c000..c004)")
    ap.add_argument("--matrices_dir", default=None,
                    help="Directory with 'alphabet' and 'sparse' files "
                         "(default: <data_dir>/Matrices)")
    ap.add_argument("--alphabet", default=None, help="Path to alphabet file (overrides matrices_dir)")
    ap.add_argument("--sparse", default=None, help="Path to encoding matrix file (overrides matrices_dir)")
    ap.add_argument("--alleles", default="all",
                    help="'all' to auto-discover, or a comma list e.g. A0101,A0201")
    ap.add_argument("--lambdas", default="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30",
                    help="Comma-separated lambda values (the complexity axis)")
    ap.add_argument("--epsilons", default="0.05",
                    help="Comma-separated learning-rate values")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers (across alleles)")
    ap.add_argument("--out_prefix", default="cv")
    args = ap.parse_args()

    mdir = args.matrices_dir or os.path.join(args.data_dir, "Matrices")
    alphabet_path = args.alphabet or os.path.join(mdir, "alphabet")
    sparse_path = args.sparse or os.path.join(mdir, "sparse")
    alphabet, scheme = smm.load_encoding(alphabet_path, sparse_path)

    alleles = (discover_alleles(args.data_dir) if args.alleles == "all"
               else [a.strip() for a in args.alleles.split(",") if a.strip()])
    if not alleles:
        sys.exit(f"No alleles found under {args.data_dir}")

    grid = list(itertools.product(_floats(args.lambdas), _floats(args.epsilons),
                                  [args.epochs]))

    print(f"Alleles: {len(alleles)} | grid: {len(grid)} HP settings "
          f"({len(_floats(args.lambdas))} lambda x {len(_floats(args.epsilons))} epsilon, "
          f"epochs={args.epochs}) | jobs={args.jobs}", file=sys.stderr)

    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            results = pool.starmap(
                run_allele,
                [(a, args.data_dir, scheme, alphabet, grid, args.seed) for a in alleles])
    else:
        results = []
        for a in alleles:
            results.append(run_allele(a, args.data_dir, scheme, alphabet, grid, args.seed))
            print(f"  done: {a}", file=sys.stderr)

    write_outputs(results, grid, args.out_prefix)
    write_predictions(results, args.out_prefix)
    print_summary(results)
    print(f"\nWrote {args.out_prefix}_fake_grid.csv, {args.out_prefix}_summary.csv "
         f"and {args.out_prefix}_predictions.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
