"""Run both baselines on the validation split for each size and seed."""

import argparse
from pathlib import Path

import pandas as pd

from baselines import build_baselines
from prepare import RUN_SEEDS, SAMPLE_SIZES, sample_training, split_tickets


def run_baselines() -> pd.DataFrame:
    """Collect validation scores without evaluating the held-out test set."""
    train, validation, _ = split_tickets()
    results = []
    for size in SAMPLE_SIZES:
        for seed in RUN_SEEDS:
            sample = sample_training(train, size, seed)
            for method, model in build_baselines(seed).items():
                model.fit(sample["text"], sample["label"])
                results.append(
                    {
                        "method": method,
                        "training_size": size,
                        "seed": seed,
                        "validation_accuracy": model.score(
                            validation["text"], validation["label"]
                        ),
                    }
                )
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/baselines.csv"))
    args = parser.parse_args()
    results = run_baselines()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary = results.groupby(["training_size", "method"])["validation_accuracy"].agg(
        ["mean", "min", "max"]
    )
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
