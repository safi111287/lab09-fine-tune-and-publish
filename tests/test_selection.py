import pandas as pd

from run_experiments import select_results


def test_selects_one_epoch_by_mean_instead_of_the_best_individual_seed():
    validation = pd.DataFrame(
        [
            {
                "method": method,
                "training_size": 50,
                "seed": seed,
                "epoch": epoch,
                "validation_accuracy": score,
            }
            for method, epoch, scores in [
                ("full", 1, [0.75, 0.75, 0.75]),
                ("full", 2, [0.99, 0.20, 0.75]),
                ("full", 3, [0.75, 0.75, 0.75]),
                ("most_common", None, [0.28, 0.28, 0.28]),
            ]
            for seed, score in enumerate(scores, start=1)
        ]
    )
    selected = select_results(validation)
    full = selected[selected.method.eq("full")]
    assert len(full) == 3
    assert set(full.epoch) == {1}
    assert set(full.seed) == {1, 2, 3}
    assert len(selected[selected.method.eq("most_common")]) == 3
