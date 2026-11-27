from copy import deepcopy

import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from finetune import (
    LABEL_TO_ID,
    MAX_LENGTH,
    encode,
    fit,
    make_loader,
    predict,
    save_checkpoint,
    seed_everything,
    set_trainability,
)
from share import model_card
from tickets_data import LABELS

DEVICE = torch.device("cpu")
WORDS = ["fee", "wheel", "dock", "login"]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    torch.set_num_threads(1)
    vocab = tmp_path_factory.mktemp("tokenizer") / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]),
        encoding="utf-8",
    )
    return DistilBertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)


def tiny_model(variant):
    seed_everything(31)
    config = DistilBertConfig(
        vocab_size=9,
        dim=24,
        hidden_dim=32,
        n_heads=2,
        n_layers=1,
        num_labels=4,
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
    )
    model = DistilBertForSequenceClassification(config)
    set_trainability(model, variant)
    return model


def minibatch(tokenizer):
    return make_loader(tokenizer, encode(tokenizer, WORDS, LABELS), 4, seed=31)


def test_tokenization_truncates_and_preserves_labels(tokenizer):
    rows = encode(tokenizer, ["fee " * 100, "dock"], ["billing", "station"])
    assert len(rows[0]["input_ids"]) == MAX_LENGTH
    assert len(rows[1]["input_ids"]) < MAX_LENGTH
    assert [row["labels"] for row in rows] == [
        LABEL_TO_ID["billing"],
        LABEL_TO_ID["station"],
    ]
    loader = make_loader(tokenizer, rows[1:], 1)
    assert next(iter(loader))["input_ids"].shape[-1] < MAX_LENGTH


def test_frozen_body_does_not_change_but_head_learns(tokenizer):
    model = tiny_model("head_only")
    body_before = deepcopy(model.distilbert.state_dict())
    head_before = model.classifier.weight.clone()
    assert not any(param.requires_grad for param in model.distilbert.parameters())
    assert all(param.requires_grad for param in model.classifier.parameters())
    fit(model, minibatch(tokenizer), DEVICE, "head_only", 0.01, 1)
    assert all(
        torch.equal(value, body_before[name])
        for name, value in model.distilbert.state_dict().items()
    )
    assert not torch.equal(model.classifier.weight, head_before)


def test_full_training_changes_body_and_same_seed_repeats(tokenizer):
    first = tiny_model("full")
    original = first.distilbert.embeddings.word_embeddings.weight.clone()
    fit(first, minibatch(tokenizer), DEVICE, "full", 0.01, 1)
    second = tiny_model("full")
    fit(second, minibatch(tokenizer), DEVICE, "full", 0.01, 1)
    assert not torch.equal(first.distilbert.embeddings.word_embeddings.weight, original)
    assert all(
        torch.equal(value, second.state_dict()[name])
        for name, value in first.state_dict().items()
    )


def test_model_can_memorize_four_tickets(tokenizer):
    model = tiny_model("full")
    # One short epoch over copies of a single four-ticket batch isolates learnability.
    repeated = make_loader(tokenizer, encode(tokenizer, WORDS * 65, LABELS * 65), 4)
    fit(model, repeated, DEVICE, "full", 0.012, 1)
    assert predict(model, minibatch(tokenizer), DEVICE) == LABELS


def test_saved_model_reloads_identical_logits(tmp_path, tokenizer):
    model = tiny_model("full")
    model.eval()
    fit(model, minibatch(tokenizer), DEVICE, "full", 0.01, 1)
    save_checkpoint(model, tmp_path)
    tokenizer.save_pretrained(tmp_path)
    restored = AutoModelForSequenceClassification.from_pretrained(tmp_path)
    batch = tokenizer(WORDS, padding=True, return_tensors="pt")
    with torch.inference_mode():
        torch.testing.assert_close(
            model.eval()(**batch).logits, restored(**batch).logits
        )
    assert restored.config.id2label == dict(enumerate(LABELS))


def test_model_card_uses_the_required_headings_and_metrics():
    scores = {"accuracy": 0.875, "macro_f1": 0.868}
    card = model_card(
        {
            "model": "full",
            "training_size": 300,
            "seed": 22,
            "epoch": 2,
            "learning_rate": 0.00005,
            "scores": {
                "test": dict.fromkeys(("full", "most_common", "tfidf_logistic"), scores)
            },
        }
    )
    assert card.startswith("---\nlicense:")
    assert all(
        f"## {name}" in card
        for name in ("Intended use", "Training data", "Results", "Limitations")
    )
    assert "300" in card and "0.875" in card and "tfidf logistic" in card


def test_app_scores_four_queues_and_empty_input(tmp_path, tokenizer, monkeypatch):
    from space.app import classify

    save_checkpoint(tiny_model("head_only"), tmp_path)
    tokenizer.save_pretrained(tmp_path)
    monkeypatch.setenv("MODEL", str(tmp_path))
    scores, message = classify("My wheel will not turn.")
    assert set(scores) == set(LABELS) and sum(scores.values()) == pytest.approx(1)
    assert max(scores, key=scores.get) in message
    assert classify("  ") == (None, "Enter a support message to see the queue scores.")
