"""Train DistilBERT while keeping held-out evaluation separate."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from tickets_data import LABELS

BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MAX_LENGTH = 64
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
RATES = {"head_only": 1e-3, "full": 5e-5}
EPOCHS = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def model_source():
    path = os.environ.get("BASE_MODEL_DIR")
    return (path, {}) if path else (BASE_MODEL, {"revision": BASE_REVISION})


def make_model(variant, seed):
    seed_everything(seed)
    source, options = model_source()
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label=dict(enumerate(LABELS)),
        **options,
    )
    set_trainability(model, variant)
    return model


def set_trainability(model, variant):
    if variant not in RATES:
        raise ValueError(f"Unknown training variant: {variant}")
    model.distilbert.requires_grad_(variant == "full")


def encode(tokenizer, texts, labels=None):
    tokens = tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)
    rows = [
        dict(zip(tokens, values, strict=True))
        for values in zip(*tokens.values(), strict=True)
    ]
    if labels is not None:
        for row, label in zip(rows, labels, strict=True):
            row["labels"] = LABEL_TO_ID[label]
    return rows


def make_loader(tokenizer, rows, batch_size, *, shuffle=False, seed=0):
    return DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def predict(model, loader, device):
    model.eval()
    predicted = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            predicted.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    return [LABELS[value] for value in predicted]


def fit(model, loader, device, variant, learning_rate, epochs, on_epoch=None):
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    for epoch in range(1, epochs + 1):
        model.train()
        if variant == "head_only":
            model.distilbert.eval()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            size = batch["labels"].shape[0]
            total_loss += loss.item() * size
            total_rows += size
        if on_epoch is not None:
            on_epoch(epoch, total_loss / total_rows, model)


def get_tokenizer():
    source, options = model_source()
    return AutoTokenizer.from_pretrained(source, **options)


def save_checkpoint(model, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
