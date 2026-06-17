import os
import math
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn as nn

CORE_LEN = 9
OUTPUT_SIZE = 1
N_EPOCHS = 2000
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


def train_network(net, x_train, y_train, pep_idx_train, optimizer):
    net.train()
    loss_fn = nn.MSELoss()

    preds = net(x_train, pep_idx_train)
    loss = loss_fn(preds, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval_network(net, x_valid, y_valid, pep_idx_valid):
    net.eval()
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        preds = net(x_valid, pep_idx_valid)
        loss = loss_fn(preds, y_valid)

    return loss.item()


def save_ffnn_model(filepath, model):
    if not filepath.endswith(".pt"):
        filepath = filepath + ".pt"

    dict_to_save = {
        "input_size": model.in_layer.in_features,
        "hidden_size": model.in_layer.out_features,
        "output_size": model.out_layer.out_features,
        "state_dict": model.state_dict()
    }

    torch.save(dict_to_save, filepath)
    print(f"Saved FFNN model at {filepath}")


def main():
    parser = ArgumentParser(description="Train an NNAlign-like FFNN model")
    parser.add_argument("--train_data", required=True, type=str, help="Path to training data file")
    parser.add_argument("--valid_data", required=True, type=str, help="Path to validation data file")
    parser.add_argument("--hidden_size", required=True, type=int, help="Number of hidden layer neurons")
    parser.add_argument("--learning_rate", required=True, type=float, help="Learning rate")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to output directory")
    parser.add_argument("--blosum_file", default=DEFAULT_BLOSUM, type=str, help="Path to BLOSUM file")
    parser.add_argument("--epochs", default=N_EPOCHS, type=int, help="Number of training epochs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)

    train_raw = load_peptide_target(args.train_data)
    valid_raw = load_peptide_target(args.valid_data)

    x_train, y_train, pep_idx_train = encode_peptides(train_raw, args.blosum_file, CORE_LEN)
    x_valid, y_valid, pep_idx_valid = encode_peptides(valid_raw, args.blosum_file, CORE_LEN)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)

    input_size = x_train.shape[1]
    model = SimpleFFNN(input_size=input_size, hidden_size=args.hidden_size, output_size=OUTPUT_SIZE)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train_losses = []
    valid_losses = []

    for epoch in range(args.epochs):
        train_loss = train_network(model, x_train, y_train, pep_idx_train, optimizer)
        valid_loss = eval_network(model, x_valid, y_valid, pep_idx_valid)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if args.epochs <= 10 or epoch % max(1, math.ceil(0.05 * args.epochs)) == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")

    model_path = os.path.join(args.output_dir, "nnalign_model.pt")
    save_ffnn_model(model_path, model)

    loss_path = os.path.join(args.output_dir, "losses.csv")
    pd.DataFrame({
        "epoch": list(range(args.epochs)),
        "train_loss": train_losses,
        "valid_loss": valid_losses
    }).to_csv(loss_path, index=False)


if __name__ == "__main__":
    main()
