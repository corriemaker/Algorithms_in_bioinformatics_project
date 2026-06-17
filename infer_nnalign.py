import os
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn as nn

CORE_LEN = 9
DEFAULT_BLOSUM = "BLOSUM50"


def load_blosum(filename):
    aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep=r"\s+", comment="#", index_col=0)
    blosum_matrix = torch.tensor(df.loc[aa, aa].to_numpy(), dtype=torch.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(aa)}
    return blosum_matrix, aa_to_idx


def load_peptide_target(filename):
    df = pd.read_csv(filename, sep=r"\s+", usecols=[0, 1], names=["peptide", "target"])
    return df.sort_values(by="target", ascending=False).reset_index(drop=True)


def encode_peptides(x_in, blosum_file, core_len=9):
    blosum50, aa_to_idx = load_blosum(blosum_file)

    x_idx = []
    pep_idx = []

    for i, peptide in enumerate(x_in["peptide"]):
        for j in range(len(peptide) - core_len + 1):
            core = peptide[j:j + core_len]
            x_idx.append([aa_to_idx.get(aa, -1) for aa in core])
            pep_idx.append(i)

    x_idx = torch.tensor(x_idx, dtype=torch.long)
    x_out = blosum50[x_idx]
    y_out = torch.tensor(x_in["target"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    pep_idx = torch.tensor(pep_idx, dtype=torch.long).reshape(-1, 1)

    return x_out, y_out, pep_idx


def xavier_initialization_normal(layer_weights):
    return nn.init.xavier_normal_(layer_weights)


class SimpleFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initialization_function=xavier_initialization_normal):
        super().__init__()
        self.in_layer = nn.Linear(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Sigmoid()

        initialization_function(self.in_layer.weight)
        initialization_function(self.out_layer.weight)

    def forward(self, x, pep_idx):
        z1 = self.in_layer(x)
        a1 = self.hidden_act(z1)
        z2 = self.out_layer(a1)
        a2 = self.out_act(z2)

        n_peptides = int(pep_idx.max().item()) + 1

        a2_best = torch.full(
            size=(n_peptides, a2.shape[1]),
            fill_value=-float("inf"),
            dtype=a2.dtype,
            device=a2.device
        )

        a2_best.scatter_reduce_(
            dim=0,
            index=pep_idx.expand(-1, a2.shape[1]),
            src=a2,
            reduce="amax",
            include_self=True
        )

        return a2_best


def load_ffnn_model(filepath, model=None):
    loaded_dict = torch.load(filepath, map_location="cpu")

    if model is None:
        model = SimpleFFNN(
            input_size=loaded_dict["input_size"],
            hidden_size=loaded_dict["hidden_size"],
            output_size=loaded_dict["output_size"]
        )

    assert model.in_layer.in_features == loaded_dict["input_size"]
    assert model.in_layer.out_features == loaded_dict["hidden_size"]
    assert model.out_layer.out_features == loaded_dict["output_size"]

    model.load_state_dict(loaded_dict["state_dict"])
    return model


def main():
    parser = ArgumentParser(description="Run inference with a trained NNAlign-like FFNN model")
    parser.add_argument("--inference_data", required=True, type=str, help="Path to inference data file")
    parser.add_argument("--parameter_file", required=True, type=str, help="Path to trained model parameter file")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to output directory")
    parser.add_argument("--blosum_file", default=DEFAULT_BLOSUM, type=str, help="Path to BLOSUM file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer_raw = load_peptide_target(args.inference_data)
    x_infer, y_infer, pep_idx_infer = encode_peptides(infer_raw, args.blosum_file, CORE_LEN)
    x_infer = x_infer.reshape(x_infer.shape[0], -1)

    model = load_ffnn_model(args.parameter_file)
    model.eval()

    with torch.no_grad():
        pred_scores = model(x_infer, pep_idx_infer).squeeze(1).cpu().numpy()

    out_df = infer_raw.copy()
    out_df["prediction"] = pred_scores
    out_path = os.path.join(args.output_dir, "predictions.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
