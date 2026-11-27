---
title: Lakeshore Ticket Sorter
emoji: 🚲
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
---

Suggests a support queue for a Lakeshore Bike Share ticket and shows scores for
billing, bike fault, station, and account. It uses a DistilBERT classifier
trained on invented tickets and does not route real cases automatically.
