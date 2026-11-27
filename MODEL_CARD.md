---
license: apache-2.0
base_model: distilbert/distilbert-base-uncased
pipeline_tag: text-classification
language: en
---

# Lakeshore Ticket Sorter

## Intended use

Suggest a queue for short English bike-share support messages: billing,
bike_fault, station, or account. Reviewers should confirm the suggestion.

## Training data

Trained on 300 stratified examples from an invented
Lakeshore Bike Share dataset. The generated dataset was split 60/20/20
before fitting. Validation chose the epoch and fine-tuning variant.
The published checkpoint uses full fine-tuning, sampling seed
22, 3 epochs and a learning rate of
5e-05.

## Results

All three methods used the same final training examples and held-out
generated test rows.

| Method | Accuracy | Macro F1 |
| :--- | ---: | ---: |
| full | 0.900 | 0.902 |
| most common | 0.287 | 0.112 |
| tfidf logistic | 0.871 | 0.873 |

## Limitations

The training text was invented and repeats templates. Some messages
are vague or mix topics. Scores are not calibrated confidence estimates.
Performance on real customers, other languages, and unusual wording has
not been established.
