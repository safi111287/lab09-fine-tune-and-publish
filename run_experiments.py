"""Run a validation grid, then evaluate the selected checkpoints on held-out data."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baselines import build_baselines
from evaluate import metrics, plot_confusion, plot_learning_curve
from finetune import (
    EPOCHS,
    RATES,
    encode,
    fit,
    get_tokenizer,
    make_loader,
    make_model,
    predict,
    save_checkpoint,
)
from inspect_data import write_data_profile
from prepare import RUN_SEEDS, SAMPLE_SIZES, sample_training, split_tickets
from share import write_model_card
from tickets_data import LABELS


def load_own(path):
    rows = pd.read_csv(path, keep_default_na=False)
    if (
        rows.columns.tolist() != ["text", "label"]
        or rows["label"].value_counts().to_dict() != dict.fromkeys(LABELS, 10)
        or rows["text"].str.strip().eq("").any()
        or rows["text"].str.strip().str.casefold().duplicated().any()
    ):
        raise ValueError("Use 40 distinct nonempty messages: ten for each queue.")
    generated = set(pd.concat(split_tickets())["text"].str.strip().str.casefold())
    if set(rows["text"].str.strip().str.casefold()) & generated:
        raise ValueError("The independent messages must not copy generated tickets.")
    return rows


def run_grid(reports, checkpoints, device, batch_size):
    train, validation, _ = split_tickets()
    tokenizer = get_tokenizer()
    val_loader = make_loader(tokenizer, encode(tokenizer, validation.text), batch_size)
    results, history = [], []
    for size in SAMPLE_SIZES:
        for seed in RUN_SEEDS:
            sample = sample_training(train, size, seed)
            for method, baseline in build_baselines(seed).items():
                baseline.fit(sample.text, sample.label)
                score = metrics(validation.label, baseline.predict(validation.text))
                results.append(
                    dict(
                        method=method,
                        training_size=size,
                        seed=seed,
                        learning_rate=None,
                        epoch=None,
                        validation_accuracy=score["accuracy"],
                        validation_macro_f1=score["macro_f1"],
                    )
                )
            for variant, rate in RATES.items():
                model = make_model(variant, seed)
                loader = make_loader(
                    tokenizer,
                    encode(tokenizer, sample.text, sample.label),
                    batch_size,
                    shuffle=True,
                    seed=seed,
                )
                run = []

                def log(
                    epoch,
                    loss,
                    trained,
                    *,
                    variant=variant,
                    size=size,
                    seed=seed,
                    rate=rate,
                    run=run,
                ):
                    score = metrics(
                        validation.label, predict(trained, val_loader, device)
                    )
                    row = dict(
                        method=variant,
                        training_size=size,
                        seed=seed,
                        learning_rate=rate,
                        epoch=epoch,
                        training_loss=loss,
                        validation_accuracy=score["accuracy"],
                        validation_macro_f1=score["macro_f1"],
                    )
                    run.append(row)
                    history.append(row)
                    print(row, flush=True)
                    if size == max(SAMPLE_SIZES) and seed == sorted(RUN_SEEDS)[1]:
                        folder = checkpoints / f"{variant}-{seed}-{epoch}"
                        save_checkpoint(trained, folder)
                    pd.DataFrame(history).to_csv(reports / "epochs.csv", index=False)

                fit(model, loader, device, variant, rate, EPOCHS, log)
                results.extend(run)
    return pd.DataFrame(results)


def select_results(raw):
    keys = ["method", "training_size"]
    neural = raw.loc[raw["epoch"].notna()]
    scores = neural.groupby(keys + ["epoch"], as_index=False)[
        "validation_accuracy"
    ].mean()
    epochs = scores.sort_values(
        ["validation_accuracy", "epoch"], ascending=[False, True]
    ).drop_duplicates(keys)[keys + ["epoch"]]
    selected = neural.merge(epochs, on=keys + ["epoch"])
    baselines = raw.loc[raw["epoch"].isna()]
    return pd.concat([baselines, selected], ignore_index=True)


def evaluate_final(selected, own, reports, checkpoints, model_dir, device, batch_size):
    train, _, test = split_tickets()
    tokenizer = get_tokenizer()
    largest = selected.loc[selected.training_size.eq(max(SAMPLE_SIZES))]
    means = largest.groupby("method").validation_accuracy.mean()
    variant = max(RATES, key=lambda method: (means[method], method == "head_only"))
    seed = sorted(RUN_SEEDS)[1]
    epoch = int(largest.loc[largest.method.eq(variant), "epoch"].iloc[0])
    sample = sample_training(train, max(SAMPLE_SIZES), seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoints / f"{variant}-{seed}-{epoch}"
    ).to(device)
    evaluation_sets = {"test": test}
    if own is not None:
        evaluation_sets["own_messages"] = own
    results = {
        "model": variant,
        "seed": seed,
        "training_size": len(sample),
        "epoch": epoch,
        "learning_rate": RATES[variant],
        "overall_recommendation_by_validation": str(means.idxmax()),
        "scores": {},
    }
    for name, rows in evaluation_sets.items():
        scores, predicted = score_rows(
            model, tokenizer, sample, rows, seed, device, batch_size
        )
        results["scores"][name] = {variant: scores.pop("distilbert"), **scores}
        if name == "test":
            plot_confusion(
                rows.label, predicted, reports / "confusion_matrix.png", variant
            )
    (reports / "final.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(model, model_dir)
    tokenizer.save_pretrained(model_dir)
    write_model_card(model_dir, reports / "final.json")


def score_rows(model, tokenizer, sample, rows, seed, device, batch_size):
    loader = make_loader(tokenizer, encode(tokenizer, rows.text), batch_size)
    predicted = predict(model, loader, device)
    scores = {"distilbert": metrics(rows.label, predicted)}
    for name, baseline in build_baselines(seed).items():
        baseline.fit(sample.text, sample.label)
        scores[name] = metrics(rows.label, baseline.predict(rows.text))
    return scores, predicted


def score_own_only(path, model_dir, rows, device, batch_size):
    result = json.loads(path.read_text(encoding="utf-8"))
    train, _, _ = split_tickets()
    seed = result["seed"]
    sample = sample_training(train, result["training_size"], seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    scores, _ = score_rows(model, tokenizer, sample, rows, seed, device, batch_size)
    result["scores"]["own_messages"] = {
        result["model"]: scores.pop("distilbert"),
        **scores,
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--threads", type=int, default=7)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validation-only", action="store_true", help="Train and validate only."
    )
    mode.add_argument(
        "--final-only",
        action="store_true",
        help="Evaluate saved checkpoints without retraining.",
    )
    mode.add_argument(
        "--own-only", action="store_true", help="Score only the independent messages."
    )
    parser.add_argument(
        "--skip-own",
        action="store_true",
        help="Evaluate the generated test set without the independent CSV.",
    )
    parser.add_argument("--own", type=Path, default=Path("data/own_messages.csv"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.reports.mkdir(parents=True, exist_ok=True)
    results_path = args.reports / "results.csv"
    if args.own_only:
        score_own_only(
            args.reports / "final.json",
            args.model_dir,
            load_own(args.own),
            device,
            args.batch_size,
        )
        return
    own = None if args.validation_only or args.skip_own else load_own(args.own)
    if args.final_only:
        results = pd.read_csv(results_path)
    else:
        results = run_grid(args.reports, args.checkpoints, device, args.batch_size)
        results.to_csv(results_path, index=False)
    write_data_profile(args.reports / "data_profile.json")
    selected = select_results(results)
    selected.to_csv(args.reports / "selected_results.csv", index=False)
    plot_learning_curve(selected, args.reports / "learning_curve.png")
    if not args.validation_only:
        evaluate_final(
            selected,
            own,
            args.reports,
            args.checkpoints,
            args.model_dir,
            device,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
