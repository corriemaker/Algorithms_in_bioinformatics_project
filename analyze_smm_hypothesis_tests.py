import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "cv_summary.csv"
ALLELE_OUTPUT_PATH = ROOT / "smw_cv_method_test_allele_differences.csv"
SUMMARY_OUTPUT_PATH = ROOT / "smw_cv_method_test_summary.csv"


def load_summary_rows():
    with INPUT_PATH.open() as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return sum(values) / len(values)


def median(values):
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def weighted_mean(values, weights):
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def sign_test_p_value(wins, losses):
    n_eff = wins + losses
    if n_eff == 0:
        return 1.0

    tail = 0.0
    for i in range(wins, n_eff + 1):
        tail += math.comb(n_eff, i)
    return tail / (2 ** n_eff)


def classify_pcc(diff):
    if diff > 0:
        return "win"
    if diff < 0:
        return "loss"
    return "tie"


def classify_mse(diff):
    if diff < 0:
        return "win"
    if diff > 0:
        return "loss"
    return "tie"


def build_allele_rows(summary_rows):
    allele_rows = []

    for row in summary_rows:
        fake_best_pcc = float(row["fake_best_pcc"])
        true_pcc = float(row["true_pcc"])
        fake_best_mse = float(row["fake_best_mse"])
        true_mse = float(row["true_mse"])

        pcc_diff = fake_best_pcc - true_pcc
        mse_diff = fake_best_mse - true_mse

        allele_rows.append(
            {
                "allele": row["allele"],
                "n_peptides": int(row["n_peptides"]),
                "fake_best_pcc": fake_best_pcc,
                "true_pcc": true_pcc,
                "pcc_diff": pcc_diff,
                "fake_best_mse": fake_best_mse,
                "true_mse": true_mse,
                "mse_diff": mse_diff,
                "pcc_direction": classify_pcc(pcc_diff),
                "mse_direction": classify_mse(mse_diff),
            }
        )

    return allele_rows


def summarize_metric(allele_rows, metric_name, diff_key, direction_key, alternative):
    diffs = [row[diff_key] for row in allele_rows]
    weights = [row["n_peptides"] for row in allele_rows]
    directions = [row[direction_key] for row in allele_rows]

    wins = sum(direction == "win" for direction in directions)
    ties = sum(direction == "tie" for direction in directions)
    losses = sum(direction == "loss" for direction in directions)
    n_total = len(allele_rows)
    n_effective = wins + losses

    return {
        "metric": metric_name,
        "alternative": alternative,
        "n_total": n_total,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "n_effective": n_effective,
        "p_value": sign_test_p_value(wins, losses),
        "mean_diff": mean(diffs),
        "median_diff": median(diffs),
        "weighted_mean_diff": weighted_mean(diffs, weights),
    }


def write_allele_output(rows):
    with ALLELE_OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "allele",
                "n_peptides",
                "fake_best_pcc",
                "true_pcc",
                "pcc_diff",
                "fake_best_mse",
                "true_mse",
                "mse_diff",
                "pcc_direction",
                "mse_direction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_output(rows):
    with SUMMARY_OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "alternative",
                "n_total",
                "wins",
                "ties",
                "losses",
                "n_effective",
                "p_value",
                "mean_diff",
                "median_diff",
                "weighted_mean_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_metric_summary(summary):
    print(f"{summary['metric']} results")
    print(f"  alternative: {summary['alternative']}")
    print(f"  n_total: {summary['n_total']}")
    print(f"  wins: {summary['wins']}")
    print(f"  ties: {summary['ties']}")
    print(f"  losses: {summary['losses']}")
    print(f"  n_effective: {summary['n_effective']}")
    print(f"  p_value: {summary['p_value']}")
    print(f"  mean_diff: {summary['mean_diff']}")
    print(f"  median_diff: {summary['median_diff']}")
    print(f"  weighted_mean_diff: {summary['weighted_mean_diff']}")
    print()


def main():
    summary_rows = load_summary_rows()
    allele_rows = build_allele_rows(summary_rows)

    pcc_summary = summarize_metric(
        allele_rows,
        metric_name="PCC",
        diff_key="pcc_diff",
        direction_key="pcc_direction",
        alternative="fake_greater_than_true",
    )
    mse_summary = summarize_metric(
        allele_rows,
        metric_name="MSE",
        diff_key="mse_diff",
        direction_key="mse_direction",
        alternative="fake_less_than_true",
    )

    write_allele_output(allele_rows)
    write_summary_output([pcc_summary, mse_summary])

    print_metric_summary(pcc_summary)
    print_metric_summary(mse_summary)


if __name__ == "__main__":
    main()
