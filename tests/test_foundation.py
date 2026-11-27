import pandas as pd
import pytest

from baselines import build_baselines
from prepare import sample_training, split_tickets
from tickets_data import LABELS


@pytest.mark.parametrize("seed", (7, 19, 2026))
def test_split_sizes_and_disjoint_ids(seed):
    parts = split_tickets(seed)
    assert [len(part) for part in parts] == [720, 240, 240]
    train, validation, test = (set(part["ticket_id"]) for part in parts)
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(range(70000, 71200))
    for part in parts:
        assert set(part["label"]) == set(LABELS)


@pytest.mark.parametrize("size", (4, 50, 150, 300, 720))
def test_stratified_sample_stays_inside_train(size):
    train, _, _ = split_tickets()
    sample = sample_training(train, size, seed=11)
    assert len(sample) == sample["ticket_id"].nunique() == size
    assert set(sample["label"]) == set(LABELS)
    assert set(sample["ticket_id"]) <= set(train["ticket_id"])
    pd.testing.assert_frame_equal(sample, sample_training(train, size, seed=11))


def test_invalid_sample_size_is_rejected():
    train, _, _ = split_tickets()
    with pytest.raises(ValueError, match="Sample size"):
        sample_training(train, 3, seed=11)


def test_tfidf_beats_majority_on_validation():
    train, validation, _ = split_tickets()
    sample = sample_training(train, 300, seed=11)
    scores = {
        name: model.fit(sample["text"], sample["label"]).score(
            validation["text"], validation["label"]
        )
        for name, model in build_baselines(11).items()
    }
    assert scores["tfidf_logistic"] > scores["most_common"]
