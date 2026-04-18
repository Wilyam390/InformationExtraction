# DocClassify — Document Classification & Invoice Extraction

**IE University · AI: Statistical Learning and Prediction · Group Work**

---

## Overview

A traditional ML pipeline (no generative AI) that:
1. **Classifies** documents into 4 categories: `invoice`, `email`, `scientific_report`, `letter`
2. **Extracts** 6 structured fields from invoices (number, dates, issuer, recipient, total)

| Model | Accuracy | Method |
|---|---|---|
| LinearSVC + TF-IDF | **99.97%** | Word (1-2) + Char (3-5) n-grams |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# Optional: add local Kaggle invoice images for real OCR training data
# Place extracted images under:
#   data/raw/invoice_kaggle_images/
# or set:
#   export INVOICE_IMAGE_DIR="/absolute/path/to/extracted/images"
# Limit how many local images are OCR-processed per run (default: 1500)
#   export LOCAL_INVOICE_IMAGE_MAX=1500

# 2. Build dataset (downloads real data + generates synthetic samples)
python3 src/build_dataset.py

# 3. Train classifier
python3 src/train.py

# 4. Launch demo UI
python3 demo/app.py   # → http://localhost:7860

# 5. CLI inference
python3 src/predict.py "Invoice Number: INV-001 ..."
python3 src/predict.py path/to/document.pdf
```

---

## Project Structure

```
InformationExtraction/
├── data/
│   ├── raw/                  # (auto-populated by build_dataset.py)
│   │   └── invoice_kaggle_images/   # optional local Kaggle invoice images (OCR source)
│   └── processed/
│       ├── train.csv         # 19,001 rows
│       └── test.csv          #  3,125 rows
├── src/
│   ├── build_dataset.py      # Dataset pipeline
│   ├── train.py              # Train & evaluate classifier
│   ├── extract.py            # Regex invoice field extractor
│   └── predict.py            # End-to-end inference
├── demo/
│   └── app.py                # Gradio live demo
├── models/                   # Saved artefacts (after training)
│   ├── word_vectorizer.joblib
│   ├── char_vectorizer.joblib
│   ├── classifier.joblib
│   └── meta.json
├── requirements.txt
└── README.md
```

---

## Dataset

| Label | Train | Test | Real Sources |
|---|---|---|---|
| `invoice` | 4,001 | 125 | Synthetic (diverse: standard, freelancer, utility, B2B formats) |
| `email` | 5,000 | 1,000 | **Real** — Enron corpus (SetFit/enron_spam) |
| `scientific_report` | 5,000 | 1,000 | **Real** — ArXiv abstracts (ccdv/arxiv-classification) |
| `letter` | 5,000 | 1,000 | **Real** — CUAD legal contracts (dvgodoy) |
| **TOTAL** | **19,001** | **3,125** | — |

**Key design decisions:**
- Train/test splits are drawn from disjoint real-data subsets (no leakage)
- Synthetic invoices cover 4 layout formats: standard table, freelancer, utility bill, B2B/medical
- ~34% synthetic in training adds format diversity without distorting the distribution

---

## Classification Pipeline

```
Raw text
   │
   ▼
TF-IDF Word n-grams (1–2) ──┐
                             ├── hstack → LinearSVC → label
TF-IDF Char n-grams (3–5)  ──┘
```

**Why this approach?**
- **Word n-grams** capture vocabulary patterns (`"invoice number"`, `"abstract"`, `"sincerely"`)
- **Char n-grams** capture format/punctuation cues (`"INV-"`, `"@"`, `"USD"`)
- **LinearSVC** is fast, memory-efficient and excels at high-dimensional sparse text

---

## Invoice Extraction

Regex-based extractor with multiple patterns per field, ordered most-specific → most-general:

| Field | Example Output |
|---|---|
| `invoice_number` | `INV-4821` |
| `invoice_date` | `2024-03-15` |
| `due_date` | `2024-04-15` |
| `issuer_name` | `Acme Corp` |
| `recipient_name` | `John Smith` |
| `total_amount` | `4200.00` |

Dates are normalised to ISO 8601 (`YYYY-MM-DD`). Amounts are stripped of currency symbols.

---

## Python API

```python
from src.predict import predict

# From raw text
result = predict("INVOICE\nInvoice Number: INV-001\n...")

# From file
result = predict("path/to/invoice.pdf")

print(result["label"])       # "invoice"
print(result["extraction"])  # {"invoice_number": "INV-001", ...}
```

---

## Requirements

- Python ≥ 3.9
- scikit-learn, scipy, numpy, pandas
- datasets (HuggingFace) — for dataset building only
- pdfplumber — for PDF reading
- gradio — for demo UI
- transformers + torch — optional, for Donut extraction backend
