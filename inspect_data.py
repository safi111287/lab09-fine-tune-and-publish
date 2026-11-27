"""Describe overlap in wording using only the training and validation rows."""

import json
from pathlib import Path

from prepare import split_tickets
from tickets_data import _VAGUE


def write_data_profile(out):
    train, validation, _ = split_tickets()
    train_text = set(train.text.str.lower().str.strip())
    validation_text = validation.text.str.lower().str.strip()
    vague_templates = tuple(sentence.casefold() for sentence in _VAGUE)
    vague = validation_text.map(
        lambda text: any(phrase in text for phrase in vague_templates)
    )
    result = {
        "training_label_counts": train.label.value_counts().sort_index().to_dict(),
        "validation_rows": len(validation),
        "validation_exact_text_matches_with_training": int(
            validation_text.isin(train_text).sum()
        ),
        "validation_rows_with_unchanged_vague_phrase": int(vague.sum()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(write_data_profile(Path("reports/data_profile.json")), indent=2))
