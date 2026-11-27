"""Gradio interface for the Lakeshore ticket classifier."""

import os
from functools import lru_cache

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

QUEUES = ("billing", "bike_fault", "station", "account")
MAX_LENGTH = 64


@lru_cache(maxsize=1)
def load_model(source):
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source).eval()
    if set(model.config.id2label.values()) != set(QUEUES):
        raise ValueError("The model does not have the expected four queue labels.")
    return tokenizer, model


def classify(text):
    if not text or not text.strip():
        return None, "Enter a support message to see the queue scores."
    source = os.getenv("MODEL", "model")
    try:
        tokenizer, model = load_model(source)
    except (OSError, ValueError):
        return None, "The ticket routing model is unavailable. Try again later."
    inputs = tokenizer(
        text.strip(), truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0].tolist()
    scores = {
        model.config.id2label[index]: float(score)
        for index, score in enumerate(probabilities)
    }
    suggested = max(scores, key=scores.__getitem__)
    return scores, f"Suggested queue: **{suggested}**"


def build_demo():
    with gr.Blocks(title="Lakeshore Ticket Sorter") as demo:
        gr.Markdown(
            "# Lakeshore Ticket Sorter\n"
            "Enter a bike-share support message to see scores for all four queues. "
            "The model was trained on invented tickets; use its suggestion as a guide."
        )
        text = gr.Textbox(label="Support message", lines=4, max_lines=8)
        submit = gr.Button("Suggest a queue", variant="primary")
        status = gr.Markdown()
        scores = gr.Label(label="Queue scores", num_top_classes=4)
        submit.click(classify, text, [scores, status])
        text.submit(classify, text, [scores, status])
        gr.Examples(
            [
                "That trip shows two withdrawals from my card.",
                "The front fork shakes whenever I slow down.",
                "All the racks on the corner were occupied.",
                "I cannot get back into my rider profile.",
            ],
            inputs=text,
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
