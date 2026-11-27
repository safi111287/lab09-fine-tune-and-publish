"""Write evaluation figures and compute the two reported metrics."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from tickets_data import LABELS


def metrics(expected, predicted):
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=LABELS, average="macro")
        ),
    }


def plot_learning_curve(results, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    for method, rows in results.groupby("method"):
        summary = rows.groupby("training_size")["validation_accuracy"].agg(
            ["mean", "min", "max"]
        )
        x = summary.index.to_numpy()
        ax.plot(x, summary["mean"], marker="o", label=method.replace("_", " "))
        ax.fill_between(x, summary["min"], summary["max"], alpha=0.15)
    ax.set(
        xlabel="Training examples",
        ylabel="Validation accuracy",
        ylim=(0, 1.02),
        title="Lakeshore ticket routing: mean and range across three seeds",
    )
    ax.set_xticks(sorted(results["training_size"].unique()))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(expected, predicted, path, method):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        expected, predicted, labels=LABELS, cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(f"Held-out test: {method.replace('_', ' ')}")
    fig.savefig(path, dpi=160)
    plt.close(fig)
