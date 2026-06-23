import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve
from scipy.stats import pearsonr

CORELEN = 9
OUTPUTSIZE = 1


def load_blosum(blosumfile):
    """
    Load BLOSUM substitution matrix, using the same hardcoded alphabet
    as the NNAlign training code.

    Alphabet includes X as a catch-all:
      ['A','R','N','D','C','Q','E','G','H','I',
       'L','K','M','F','P','S','T','W','Y','V','X']
    """
    aa = [
        "A", "R", "N", "D", "C",
        "Q", "E", "G", "H", "I",
        "L", "K", "M", "F", "P",
        "S", "T", "W", "Y", "V",
        "X",
    ]
    df = pd.read_csv(blosumfile, sep=r"\s+", comment="#", index_col=0)
    blosum_matrix = torch.tensor(df.loc[aa, aa].to_numpy(), dtype=torch.float32)
    aa_to_idx = {aa_code: i for i, aa_code in enumerate(aa)}
    return blosum_matrix, aa_to_idx


def load_peptide_target(filename):
    """
    Load peptide and target values from a data file.
    Assumes whitespace-separated with peptide in col 0 and target in col 1.
    """
    df = pd.read_csv(filename, sep=r"\s+", usecols=[0, 1], names=["peptide", "target"])
    return df.sort_values(by="target", ascending=False).reset_index(drop=True)


def encode_peptides(xin, blosumfile, corelen=CORELEN):
    """
    Encode peptides using a sliding CORELEN window with the BLOSUM matrix.

    Returns:
      x_out  : (n_cores, CORELEN, 21) tensor
      y_out  : (n_peptides, 1) tensor of targets
      pepidx : (n_cores, 1) tensor mapping each core to a peptide index
    """
    blosum50, aa_to_idx = load_blosum(blosumfile)
    x_idx, pep_idx = [], []

    for i, peptide in enumerate(xin["peptide"]):
        if len(peptide) < corelen:
            continue
        for j in range(len(peptide) - corelen + 1):
            core = peptide[j : j + corelen]
            # Unknown aa → X row (fallback) if present, else index 0
            x_idx.append([aa_to_idx.get(aa, aa_to_idx.get("X", 0)) for aa in core])
            pep_idx.append(i)

    if not x_idx:
        raise ValueError("No valid peptide cores could be encoded.")

    x_idx = torch.tensor(x_idx, dtype=torch.long)
    x_out = blosum50[x_idx]  # (n_cores, corelen, 21)
    y_out = torch.tensor(xin["target"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    pep_idx = torch.tensor(pep_idx, dtype=torch.long).reshape(-1, 1)
    return x_out, y_out, pep_idx


class SimpleFFNN(nn.Module):
    """
    Single hidden-layer FFNN with NNAlign-style max pooling over cores.
    """

    def __init__(self, inputsize, hiddensize, outputsize):
        super().__init__()
        self.inlayer = nn.Linear(inputsize, hiddensize)
        self.outlayer = nn.Linear(hiddensize, outputsize)
        self.hiddenact = nn.ReLU()
        self.outact = nn.Sigmoid()

        nn.init.xavier_normal_(self.inlayer.weight)
        nn.init.xavier_normal_(self.outlayer.weight)

    def forward(self, x, pepidx):
        # x: (n_cores, inputsize)
        z1 = self.inlayer(x)
        a1 = self.hiddenact(z1)
        z2 = self.outlayer(a1)
        a2 = self.outact(z2)  # (n_cores, 1)

        # NNAlign: max over cores per peptide
        n_peptides = int(pepidx.max().item()) + 1
        a2best = torch.full(
            size=(n_peptides, a2.shape[1]),
            fill_value=-float("inf"),
            dtype=a2.dtype,
            device=a2.device,
        )

        a2best.scatter_reduce_(
            dim=0,
            index=pepidx.expand(-1, a2.shape[1]),
            src=a2,
            reduce="amax",
            include_self=True,
        )
        return a2best


def train_network(net, xtrain, ytrain, pepidxtrain, optimizer):
    net.train()
    lossfn = nn.MSELoss()
    preds = net(xtrain, pepidxtrain)
    loss = lossfn(preds, ytrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_network(net, xvalid, yvalid, pepidxvalid):
    net.eval()
    lossfn = nn.MSELoss()
    with torch.no_grad():
        preds = net(xvalid, pepidxvalid)
        loss = lossfn(preds, yvalid)
    return loss.item()


def train_model(
    train_files,
    valid_file,
    hiddensize,
    learningrate,
    weightdecay,
    patience,
    seed,
    blosumfile,
    epochs=2000,
    verbose=False,
):
    """
    Train a model with early stopping on given train and validation files.
    train_files: list of cXXX paths
    valid_file : single cXXX path
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load and concatenate training data
    train_parts = [load_peptide_target(f) for f in train_files]
    train_raw = pd.concat(train_parts, ignore_index=True)
    valid_raw = load_peptide_target(valid_file)

    # Encode
    xtrain, ytrain, pepidxtrain = encode_peptides(train_raw, blosumfile, CORELEN)
    xvalid, yvalid, pepidxvalid = encode_peptides(valid_raw, blosumfile, CORELEN)

    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xvalid = xvalid.reshape(xvalid.shape[0], -1)

    model = SimpleFFNN(
        inputsize=xtrain.shape[1],
        hiddensize=hiddensize,
        outputsize=OUTPUTSIZE,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learningrate, weight_decay=weightdecay
    )

    best_valid_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        train_loss = train_network(model, xtrain, ytrain, pepidxtrain, optimizer)
        valid_loss = eval_network(model, xvalid, yvalid, pepidxvalid)
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss}
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and ((epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0):
            print(
                f"Epoch {epoch + 1}: train={train_loss:.6f}, "
                f"valid={valid_loss:.6f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1}, best epoch {best_epoch}"
                )
            break

    if best_model_state is None:
        raise RuntimeError("Training failed to produce a best model state.")

    model.load_state_dict(best_model_state)
    history_df = pd.DataFrame(history)
    return model, best_epoch, best_valid_loss, history_df


def predict(model, test_file, blosumfile):
    """
    Make peptide-level predictions on test data.
    """
    test_raw = load_peptide_target(test_file)
    xtest, ytest, pepidxtest = encode_peptides(test_raw, blosumfile, CORELEN)
    xtest = xtest.reshape(xtest.shape[0], -1)
    model.eval()
    with torch.no_grad():
        preds = model(xtest, pepidxtest).squeeze(1).cpu().numpy()
    result_df = test_raw.copy()
    result_df["prediction"] = preds
    return result_df


def compute_metrics_from_df(df, binder_threshold=0.426):
    """
    Compute PCC, MSE, AUC and ROC curve from a DataFrame with
    columns 'target' and 'prediction'.
    """
    y_true = df["target"].values
    y_pred = df["prediction"].values

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pcc = np.nan
    else:
        pcc, _ = pearsonr(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    y_bin = (y_true >= binder_threshold).astype(int)

    if len(np.unique(y_bin)) < 2:
        auc = np.nan
        fpr, tpr = None, None
    else:
        auc = roc_auc_score(y_bin, y_pred)
        fpr, tpr, _ = roc_curve(y_bin, y_pred)

    return {
        "n_peptides": len(df),
        "n_binders": int(y_bin.sum()),
        "pcc": pcc,
        "mse": mse,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
    }


def build_fold_paths(allele_dir, test_fold, valid_fold):
    """
    Build file paths for train/valid/test given an allele directory.
    Expects c000-c004 and f000-f004 inside allele_dir.
    """
    train_files = [
        os.path.join(allele_dir, f"c{i:03d}")
        for i in range(5)
        if i not in {test_fold, valid_fold}
    ]
    valid_file = os.path.join(allele_dir, f"c{valid_fold:03d}")
    test_file = os.path.join(allele_dir, f"f{test_fold:03d}")
    return train_files, valid_file, test_file


def run_non_nested_cv(
    allele_dir,
    hiddensize,
    learningrate,
    weightdecay,
    patience,
    seed,
    epochs,
    blosumfile,
    binder_threshold,
    output_dir,
    verbose=False,
):
    """
    Run 5-fold non-nested CV:
      - for each test_fold in 0..4:
        - valid_fold = (test_fold + 1) % 5
        - train on remaining 3 c-folds
        - validate on valid_fold c
        - test on f[test_fold]
    """
    os.makedirs(output_dir, exist_ok=True)
    fold_rows = []

    for test_fold in range(5):
        valid_fold = (test_fold + 1) % 5
        train_files, valid_file, test_file = build_fold_paths(
            allele_dir, test_fold, valid_fold
        )

        if verbose:
            print(
                f"\n=== Fold {test_fold} (valid {valid_fold}) ===\n"
                f"Train: {[os.path.basename(f) for f in train_files]}\n"
                f"Valid: {os.path.basename(valid_file)}\n"
                f"Test : {os.path.basename(test_file)}"
            )

        model, best_epoch, best_valid_loss, history_df = train_model(
            train_files=train_files,
            valid_file=valid_file,
            hiddensize=hiddensize,
            learningrate=learningrate,
            weightdecay=weightdecay,
            patience=patience,
            seed=seed + test_fold,
            blosumfile=blosumfile,
            epochs=epochs,
            verbose=verbose,
        )

        valid_predictions = predict(model, valid_file, blosumfile)
        test_predictions = predict(model, test_file, blosumfile)
        valid_metrics = compute_metrics_from_df(
            valid_predictions, binder_threshold=binder_threshold
        )
        test_metrics = compute_metrics_from_df(
            test_predictions, binder_threshold=binder_threshold
        )

        fold_dir = os.path.join(output_dir, f"fold_{test_fold}")
        os.makedirs(fold_dir, exist_ok=True)
        history_df.to_csv(os.path.join(fold_dir, "training_history.csv"), index=False)
        valid_predictions.to_csv(
            os.path.join(fold_dir, "validation_predictions.csv"), index=False
        )
        test_predictions.to_csv(
            os.path.join(fold_dir, "test_predictions.csv"), index=False
        )
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))

        fold_rows.append(
            {
                "test_fold": test_fold,
                "valid_fold": valid_fold,
                "train_files": ",".join(os.path.basename(x) for x in train_files),
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
                "valid_n_peptides": valid_metrics["n_peptides"],
                "valid_n_binders": valid_metrics["n_binders"],
                "valid_auc": valid_metrics["auc"],
                "valid_pcc": valid_metrics["pcc"],
                "valid_mse": valid_metrics["mse"],
                "test_n_peptides": test_metrics["n_peptides"],
                "test_n_binders": test_metrics["n_binders"],
                "test_auc": test_metrics["auc"],
                "test_pcc": test_metrics["pcc"],
                "test_mse": test_metrics["mse"],
            }
        )

    summary_df = pd.DataFrame(fold_rows)
    summary_df.to_csv(
        os.path.join(output_dir, "non_nested_cv_summary.csv"), index=False
    )

    mean_row = {
        "test_fold": "mean",
        "valid_fold": "",
        "train_files": "",
        "best_epoch": summary_df["best_epoch"].mean(),
        "best_valid_loss": summary_df["best_valid_loss"].mean(),
        "valid_n_peptides": summary_df["valid_n_peptides"].mean(),
        "valid_n_binders": summary_df["valid_n_binders"].mean(),
        "valid_auc": summary_df["valid_auc"].mean(),
        "valid_pcc": summary_df["valid_pcc"].mean(),
        "valid_mse": summary_df["valid_mse"].mean(),
        "test_n_peptides": summary_df["test_n_peptides"].mean(),
        "test_n_binders": summary_df["test_n_binders"].mean(),
        "test_auc": summary_df["test_auc"].mean(),
        "test_pcc": summary_df["test_pcc"].mean(),
        "test_mse": summary_df["test_mse"].mean(),
    }
    std_row = {
        "test_fold": "std",
        "valid_fold": "",
        "train_files": "",
        "best_epoch": summary_df["best_epoch"].std(),
        "best_valid_loss": summary_df["best_valid_loss"].std(),
        "valid_n_peptides": summary_df["valid_n_peptides"].std(),
        "valid_n_binders": summary_df["valid_n_binders"].std(),
        "valid_auc": summary_df["valid_auc"].std(),
        "valid_pcc": summary_df["valid_pcc"].std(),
        "valid_mse": summary_df["valid_mse"].std(),
        "test_n_peptides": summary_df["test_n_peptides"].std(),
        "test_n_binders": summary_df["test_n_binders"].std(),
        "test_auc": summary_df["test_auc"].std(),
        "test_pcc": summary_df["test_pcc"].std(),
        "test_mse": summary_df["test_mse"].std(),
    }
    aggregate_df = pd.concat(
        [summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )
    aggregate_df.to_csv(
        os.path.join(output_dir, "non_nested_cv_summary_with_aggregate.csv"),
        index=False,
    )
    return aggregate_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run non-nested train/validation/test CV "
            "mirroring the nested ANN notebook."
        )
    )
    parser.add_argument(
        "--allele-dir",
        required=True,
        help="Directory containing c000-c004 and f000-f004 files for one allele.",
    )
    parser.add_argument(
        "--blosum-file",
        default="resources/BLOSUM50",
        help="Path to BLOSUM matrix file.",
    )
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binder-threshold", type=float, default=0.426)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    summary = run_non_nested_cv(
        allele_dir=args.allele_dir,
        hiddensize=args.hidden_size,
        learningrate=args.learning_rate,
        weightdecay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        epochs=args.epochs,
        blosumfile=args.blosum_file,
        binder_threshold=args.binder_threshold,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    print("\nNon-nested CV completed.")
    print(
        summary[
            [
                "test_fold",
                "valid_auc",
                "valid_pcc",
                "valid_mse",
                "test_auc",
                "test_pcc",
                "test_mse",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()