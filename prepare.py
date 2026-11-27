"""Create the fixed ticket split and draw stratified training samples."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tickets_data import LABELS, make_tickets

SPLIT_SEED = 2026
SAMPLE_SIZES = (50, 150, 300)
RUN_SEEDS = (11, 22, 33)


def split_tickets(
    seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return disjoint train, validation, and test sets (60/20/20)."""
    rows = make_tickets()
    train, held_out = train_test_split(
        rows, test_size=0.4, stratify=rows["label"], random_state=seed
    )
    validation, test = train_test_split(
        held_out, test_size=0.5, stratify=held_out["label"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        (test.reset_index(drop=True)),
    )


def sample_training(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw exactly n training rows, allocating at least one to each queue."""
    if set(train["label"]) != set(LABELS):
        raise ValueError("Training rows must include all four queues.")
    if not len(LABELS) <= n <= len(train):
        raise ValueError(f"Sample size must be between {len(LABELS)} and {len(train)}.")

    groups = [train.loc[train["label"] == label] for label in LABELS]
    counts = np.array([len(group) for group in groups])
    desired = n * counts / len(train)
    allocated = np.ones(len(groups), dtype=int)
    for _ in range(n - len(groups)):
        remaining = np.where(allocated < counts, desired - allocated, -np.inf)
        allocated[int(remaining.argmax())] += 1

    rng = np.random.default_rng(seed)
    pieces = [
        group.iloc[rng.choice(len(group), size=count, replace=False)]
        for group, count in zip(groups, allocated, strict=True)
    ]
    sample = pd.concat(pieces)
    return sample.iloc[rng.permutation(n)].reset_index(drop=True)
