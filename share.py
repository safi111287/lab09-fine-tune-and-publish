"""Generate a model card from the final run's measured scores."""

import json
from pathlib import Path


def model_card(result):
    model = result["model"].replace("_", " ")
    metrics = result["scores"]["test"]
    lines = [
        "---",
        "license: apache-2.0",
        "base_model: distilbert/distilbert-base-uncased",
        "pipeline_tag: text-classification",
        "language: en",
        "---",
        "",
        "# Lakeshore Ticket Sorter",
        "",
        "## Intended use",
        "",
        "Suggest a queue for short English bike-share support messages: billing,",
        "bike_fault, station, or account. Reviewers should confirm the suggestion.",
        "",
        "## Training data",
        "",
        f"Trained on {result['training_size']} stratified examples from an invented",
        "Lakeshore Bike Share dataset. The generated dataset was split 60/20/20",
        "before fitting. Validation chose the epoch and fine-tuning variant.",
        f"The published checkpoint uses {model} fine-tuning, sampling seed",
        f"{result['seed']}, {result['epoch']} epochs and a learning rate of",
        f"{result['learning_rate']:g}.",
        "",
        "## Results",
        "",
        "All three methods used the same final training examples and held-out",
        "generated test rows.",
        "",
        "| Method | Accuracy | Macro F1 |",
        "| :--- | ---: | ---: |",
    ]
    for name, scores in metrics.items():
        label = name.replace("_", " ")
        lines.append(
            f"| {label} | {scores['accuracy']:.3f} | {scores['macro_f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "The training text was invented and repeats templates. Some messages",
            "are vague or mix topics. Scores are not calibrated confidence estimates.",
            "Performance on real customers, other languages, and unusual wording has",
            "not been established.",
            "",
        ]
    )
    return "\n".join(lines)


def write_model_card(model_dir: Path, final_json: Path):
    result = json.loads(final_json.read_text(encoding="utf-8"))
    path = model_dir / "README.md"
    path.write_text(model_card(result), encoding="utf-8")
    return path
