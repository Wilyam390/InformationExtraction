# Technical Report — DocClassify

**IE University · AI: Statistical Learning and Prediction · Group Work**

Document Classification and Information Extraction using classical Machine Learning and rule-based Natural Language Processing.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Category Selection and Justification](#2-category-selection-and-justification)
3. [Dataset](#3-dataset)
4. [Classification Pipeline](#4-classification-pipeline)
5. [Information Extraction Pipeline](#5-information-extraction-pipeline)
6. [End-to-End Inference](#6-end-to-end-inference)
7. [Results](#7-results)
8. [Testing and Quality Assurance](#8-testing-and-quality-assurance)
9. [Live Demonstration](#9-live-demonstration)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [Reproducibility](#11-reproducibility)
12. [Appendix — Repository Layout](#12-appendix--repository-layout)

---

## 1. Project Overview

The system ingests a document (raw text, PDF, or image) and performs two tasks:

1. **Classification** — assigns the document to one of four categories: `invoice`, `email`, `scientific_report`, `letter`.
2. **Information extraction** — if and only if the document is classified as an invoice, extracts six structured fields: `invoice_number`, `invoice_date`, `due_date`, `issuer_name`, `recipient_name`, `total_amount`.

All methods are **classical**: TF-IDF feature engineering, a Linear Support Vector Classifier, regular expressions for field extraction, and Tesseract OCR for scanned PDFs/images. **No generative AI is used at any stage**, in compliance with the project brief.

The design philosophy is to maximise transparency and reproducibility: every component is inspectable, deterministic (fixed random seeds), and covered by unit tests.

---

## 2. Category Selection and Justification

Four document classes were selected. The brief required invoices plus at least three additional categories, freely chosen.

| Class | Rationale |
|---|---|
| `invoice` | Required by the brief. High real-world economic value: automated invoice processing is a well-studied commercial application. |
| `email` | Highly distinctive linguistic and structural features (headers like `From:`, `To:`, `Subject:`; conversational register). Ubiquitous in business workflows. |
| `scientific_report` | Formal academic register; dense technical vocabulary; characteristic sectioning (`Abstract`, `Methodology`, `References`); frequent mathematical notation. |
| `letter` | Formal correspondence with stereotyped openings and closings (`Dear ...`, `Yours sincerely`, `Signed`); covers cover letters, complaints, contracts, tenancy agreements and government notices. |

### Key discriminating features

| Feature | invoice | email | scientific_report | letter |
|---|---|---|---|---|
| Structural markers | line items, totals, currency codes | `From:`/`To:`/`Subject:` headers | `Abstract`/`Introduction`/numbered sections | `Dear ...`, closing signature |
| Vocabulary cues | `invoice`, `total`, `due`, `VAT`, `IBAN` | `meeting`, `team`, `FYI`, `Best regards` | `we propose`, `baseline`, `experimental results` | `sincerely`, `signed`, `agreement` |
| Number density | high (monetary) | low | moderate (statistics) | low to moderate |
| Typical length | 200–800 chars | 300–1500 chars | 500–3000 chars | 400–2000 chars |

The four classes are linguistically well-separated at the lexical and n-gram levels, which justifies the effectiveness of a TF-IDF + linear classifier approach (see Section 7 for empirical validation).

---

## 3. Dataset

### Sources

Where available, **real public data** is used. Synthetic samples fill format gaps and the small invoice training set.

| Class | Train | Test | Source |
|---|---|---|---|
| `invoice` | 4,001 | 125 | Synthetic — four format templates: standard, freelancer, utility, B2B/medical. Real invoice corpora on HuggingFace (`katanaml-org/invoices-donut-data`, `mychen76/invoices-and-receipts_ocr_v1`) are attempted first but are frequently unavailable. |
| `email` | 5,000 | 1,000 | Real — Enron corpus via `SetFit/enron_spam` and `corbt/enron-emails`. |
| `scientific_report` | 5,000 | 1,000 | Real — ArXiv abstracts via `ccdv/arxiv-classification` and `gfissore/arxiv-abstracts-2021`. |
| `letter` | 5,000 | 1,000 | Real — CUAD legal contracts via `theatticusproject/cuad` and `dvgodoy/CUAD_v1_Contract_Understanding_clause_classification`. |
| **Total** | **19,001** | **3,125** | — |

### Design decisions

- **Disjoint train/test splits from real sources.** In `src/build_dataset.py`, for each class we load `n_train + n_test` real samples in a single call, shuffle, and reserve the last `n_test` exclusively for the test split. This guarantees zero sample overlap between train and test.
- **Synthetic padding on train only.** When real data is insufficient (notably for invoices), the training split is padded with synthetic samples. The test split is padded only if real data remains insufficient after train allocation.
- **Four synthetic invoice layouts.** The generator (`src/build_dataset.py`, `_invoice_block()`) produces `standard`, `freelancer`, `utility` and `b2b` formats. This provides layout diversity the classifier would otherwise miss.
- **Fixed seed (`random.seed(42)`).** Dataset generation is fully reproducible.
- **Text truncation.** All samples are truncated to 3,000 characters (1,500 for CUAD contracts, which are much longer). This caps vectorisation memory and prevents the long-tail of single documents from dominating TF-IDF statistics.

### Class imbalance

The dataset is moderately imbalanced on the train side (invoices are 21% of training, other classes ~26% each). The test set is heavily imbalanced (4% invoices, 32% each for the other three classes). We address this with `class_weight="balanced"` in the classifier (see Section 4) rather than over- or under-sampling.

---

## 4. Classification Pipeline

### 4.1 Feature engineering — dual TF-IDF

Implemented in `src/train.py`, `build_features()`. Two complementary vectorisers are fitted independently and their outputs horizontally stacked (`scipy.sparse.hstack`):

| Vectoriser | `ngram_range` | `min_df` | `max_df` | `max_features` | Purpose |
|---|---|---|---|---|---|
| Word TF-IDF | (1, 2) | 2 | 0.95 | 80,000 | Captures vocabulary patterns: `"invoice number"`, `"Dear Sir"`, `"we propose"`, `"attached"`. |
| Char `char_wb` TF-IDF | (3, 5) | 3 | 0.95 | 60,000 | Captures format and punctuation cues: `"INV-"`, `"@"`, `"USD"`, `" $"`, section markers. |

Both use `sublinear_tf=True` (logarithmic term-frequency scaling, damping the effect of very frequent terms) and `strip_accents="unicode"` (normalises diacritics for multilingual robustness).

The final feature matrix has **up to 140,000 sparse TF-IDF features**.

**Rationale for the dual representation.** Word n-grams alone miss document-format cues (e.g. the `"INV-"` prefix pattern of invoice numbers). Character n-grams alone miss high-level vocabulary cues (e.g. the scientific-report word *"proposes"*). Fusing both produces a richer representation than either alone, at minimal additional cost (linear in feature count for LinearSVC).

### 4.2 Classifier — LinearSVC

Implemented in `src/train.py`, `build_model()`.

```
LinearSVC(
C=1.0,
class_weight="balanced" (computed via sklearn.utils.class_weight),
max_iter=2000,
random_state=42,
)


**Why LinearSVC:**

1. Scales to high-dimensional sparse features efficiently (dual coordinate descent).
2. Excellent empirical performance on text classification benchmarks.
3. Memory efficient: stores a single weight vector per class.
4. Fast training (~seconds on this dataset).
5. Deterministic given `random_state`.

The choice was validated empirically by benchmarking five candidate models in `notebooks/model_comparison.ipynb` — see Section 7 for results.

### 4.3 Handling class imbalance

`class_weight="balanced"` computes per-class weights inversely proportional to class frequency. Misclassifying a rare class incurs a higher loss, pushing the decision boundary to protect minority classes.

### 4.4 Artefacts

After training, `src/train.py` persists four artefacts to `models/`:

- `word_vectorizer.joblib` (3.2 MB)
- `char_vectorizer.joblib` (2.1 MB)
- `classifier.joblib` (4.5 MB)
- `meta.json` — labels, accuracy, model name, feature description

The separation into three joblib files allows independent reuse of the vectorisers (e.g. for the exploration notebook's top-feature analysis) without reloading the full model.

---

## 5. Information Extraction Pipeline

Implemented in `src/extract.py`. Triggered only when the classifier predicts `invoice`.

### 5.1 Design philosophy

**Multiple patterns per field, ordered most-specific to most-general. First match wins.** This prioritises high-precision explicit keywords (e.g. `"Invoice Number:"`) over heuristic fallbacks, while still providing coverage for unlabelled formats.

All patterns use `re.IGNORECASE | re.MULTILINE`. A pre-processing pass collapses multiple spaces and tabs (preserving line breaks), which stabilises pattern matching across varied whitespace.

### 5.2 Field-by-field design

#### `invoice_number`

Five patterns (`src/extract.py`, `_INV_NUMBER_PATTERNS`):

1. `invoice (number|no|#|num)` → word
2. `inv(oice) [#-no.]*` → word
3. `bill (number|no|#)` → word
4. Anchored prefix formats: `INV-`, `BILL-`, `REF-`, `SIN-`, `TAX-`, `FACTURA-`
5. `receipt (number|no|#)` → word

Captured tokens allow `[\w\-/\.]` with length 3–30 (covers compound IDs like `"9BF0758D-702530"`).

#### `invoice_date` and `due_date`

A shared `_DATE_VALUE` regex captures four formats:

- ISO: `2024-03-15`
- Numeric: `15/03/2024`, `15-03-2024`, `15.03.2024`
- Long month with day first: `15 March 2024`, `15 Mar 2024`
- Long month with month first: `March 15, 2024`

Seven label-specific patterns for invoice date (e.g. `"Invoice Date"`, `"Date of issue"`, `"Billing Date"`, `"Issued on"`) and eight for due date (including `"Net 30 days"`-style inline mentions).

Raw matches are normalised to ISO 8601 (`YYYY-MM-DD`) by `_normalise_date()`. Ambiguous numeric dates (e.g. `03/04/2024`) default to day-first interpretation (European), unless the second group > 12 (which forces month-first/US interpretation).

#### `issuer_name` and `recipient_name`

Issuer patterns, in priority order:

1. Explicit label: `from|issued by|seller|vendor|supplier|biller|company`
2. Multi-word line ending with a legal suffix: `Ltd`, `LLC`, `Inc`, `GmbH`, `SL`, `SA`, `Limited`, `Corp`, `BV`, `AG`, `PLC`
3. All-caps company name on its own line

The all-caps heuristic skips generic headers (`INVOICE`, `RECEIPT`, `BILL`, `STATEMENT`, ...) via the `_GENERIC_HEADERS` denylist, avoiding the trap of capturing the document title itself.

Recipient patterns, in priority order:

1. `bill(ed) to|sold to|ship(ped) to|recipient|buyer`
2. `Dear|Attn|Attention`
3. `customer|client`
4. `To:`

Both values are truncated to the first line after match (multiline leakage protection).

#### `total_amount`

A shared `_AMOUNT_VALUE` regex captures currency-prefixed or -suffixed numeric values with comma/period thousands separators.

Six patterns, in priority order:

1. `Amount due` — most explicit
2. `Balance due`, `Outstanding due`
3. `Grand total`
4. `Total (amount) (due|payable|owed)`
5. `Total (amount):` with explicit separator
6. Line-anchored `Total` **with a negative lookahead** that excludes `Total excluding tax`, `Total excl.`, `Total before tax`, `Total net`

**Last-resort fallback.** If all keyword patterns fail, the extractor scans for every monetary-looking number in the document and returns the **largest** — under the heuristic that the total is almost always the maximum amount on an invoice (since taxes and line items add up to it).

Amounts are normalised by `_normalise_amount()`: currency symbols/codes stripped, European decimal format (`1.234,56`) detected and converted, thousands separators removed.

### 5.3 Why regex and not an ML extractor?

The project brief explicitly permits "pattern analysis, rule-based approaches with regular expressions" and forbids generative AI. Beyond the rule, regex is the appropriate tool here because:

1. **Labelled training data for field extraction is scarce and expensive.** Public invoice corpora with per-field labels exist but are small and in inconsistent formats.
2. **Regex is interpretable and debuggable.** Every extraction failure can be pinpointed to a specific pattern in `src/extract.py`. A neural extractor would be opaque.
3. **Zero dependency on labelled invoice data at inference time.** The extractor generalises to any format the pattern library covers; no retraining needed when new layouts appear.
4. **Latency.** Regex extraction runs in <1 ms per invoice, versus tens of milliseconds for an ML model and hundreds of milliseconds for transformers.

---

## 6. End-to-End Inference

Implemented in `src/predict.py`. The `predict()` function accepts raw text or a file path and dispatches to the correct reader.

### 6.1 Input handling

| Input type | Reader | Fallback |
|---|---|---|
| Raw text (>512 chars or not a valid path) | direct string use | — |
| `.txt` file | `Path.read_text()` | — |
| `.pdf` (digital/typed) | `pdfplumber` | OCR fallback if text layer < 50 chars |
| `.pdf` (scanned) | `pdf2image` + Tesseract OCR | — |
| Image (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`) | Pillow + Tesseract OCR | — |

### 6.2 OCR subsystem

Implemented in `src/predict.py`, `_ocr_pdf()`, `_ocr_image()`, `_preprocess_for_ocr()`:

- Pages rendered at 300 DPI (`pdf2image.convert_from_path(dpi=300)`).
- Pre-processing: greyscale conversion → PIL `autocontrast(cutoff=2)` → mild sharpen filter. No OpenCV dependency.
- Tesseract configuration: `--oem 3` (LSTM engine) `--psm 3` (fully automatic page segmentation).
- English language model only (`lang="eng"`).

The OCR path is automatically triggered when `pdfplumber` returns fewer than `_PDF_TEXT_MIN_CHARS = 50` characters from a PDF. This threshold reliably distinguishes typed PDFs (thousands of chars) from scanned PDFs (0–20 chars of OCR noise from embedded images).

### 6.3 Classification and extraction

1. Text is cleaned (whitespace collapse) and vectorised by the two TF-IDF vectorisers.
2. Features are horizontally stacked and passed to `LinearSVC.predict()`.
3. `LinearSVC.decision_function()` produces per-class scores (higher = more confident); these are returned verbatim for downstream UI use (confidence bar display).
4. If the predicted label is `invoice`, `extract_invoice_fields()` is invoked and its dict is added to the result.

### 6.4 Return schema

```python
{
"source": str, # file path or "<raw_text>"
"label": str, # predicted class
"confidence": {class: float, ...}, # decision-function scores
"ocr_used": bool, # true if Tesseract was invoked
"text_preview": str, # first 300 chars
"extraction": { # only if label == "invoice"
"invoice_number": str | None,
"invoice_date": str | None, # ISO 8601
"due_date": str | None, # ISO 8601
"issuer_name": str | None,
"recipient_name": str | None,
"total_amount": str | None, # currency-stripped
},
}


---

## 7. Results

### 7.1 Model comparison

Five classifiers were benchmarked on the identical TF-IDF feature matrix using 5-fold stratified cross-validation on the training set and evaluated on the held-out test set. Full experiment in `notebooks/model_comparison.ipynb`.

**5-fold cross-validation (train set, 19,001 samples)**

| Rank | Model | Mean Accuracy | Std |
|---|---|---|---|
| 1 | LinearSVC | 99.974% | 0.026% |
| 2 | VotingEnsemble | 99.963% | 0.024% |
| 3 | LogisticRegression | 99.947% | 0.032% |
| 4 | RandomForest | 99.942% | 0.012% |
| 5 | MultinomialNB | 99.942% | 0.047% |

**Held-out test set (3,125 samples)**

| Rank | Model | Test Accuracy | Macro F1 |
|---|---|---|---|
| 1 (tie) | LinearSVC | 99.968% | 0.9998 |
| 1 (tie) | VotingEnsemble | 99.968% | 0.9998 |
| 3 (tie) | MultinomialNB | 99.936% | 0.9995 |
| 3 (tie) | LogisticRegression | 99.936% | 0.9995 |
| 5 | RandomForest | 99.840% | 0.9988 |

Visual summary artefacts are saved to `reports/`:

- `reports/cv_boxplot.png` — CV accuracy distribution per model
- `reports/test_accuracy.png` — test accuracy ranking
- `reports/confusion_matrix.png` — confusion matrix for the winning model

### 7.2 Observations

1. **All five models exceed 99.8% accuracy.** The total spread between best (LinearSVC, 99.968%) and worst (RandomForest, 99.840%) is 0.13 percentage points. The task is highly separable at the TF-IDF feature level.
2. **Linear models dominate.** LinearSVC, LogisticRegression and MultinomialNB cluster at the top. This is the expected behaviour for high-dimensional sparse text representations, where linear decision boundaries generalise well.
3. **RandomForest is the weakest performer.** Tree splits select a handful of features at each node, which is inefficient across a 140,000-dimensional sparse space. Linear models assign a weight to every feature simultaneously.
4. **The Voting ensemble does not improve over LinearSVC alone.** It ties on the test set and is marginally worse in CV. When base models already agree on >99.9% of samples, ensembling has nothing to correct; it only adds engineering complexity and training time.
5. **All CV standard deviations are ≤ 0.05%.** Every model is extremely stable across folds — there is no meaningful variance risk associated with the choice.

### 7.3 Production choice justified

LinearSVC is selected for the production pipeline because:

1. Best mean CV accuracy (99.97%) and tied-best test accuracy.
2. Fastest training among the top performers (no calibration wrapper required, unlike the ensemble).
3. Minimal memory footprint — a single weight vector per class.
4. Simplicity — no probability calibration step, no ensemble-level hyperparameter tuning.

The accuracy gap between models (<0.15 pp) is much smaller than the engineering complexity gap, making LinearSVC the pragmatic choice.

### 7.4 Per-class test performance (LinearSVC)

All four classes achieve precision, recall and F1 of 1.00 on the held-out test set. Detailed per-class report is generated at the bottom of `notebooks/model_comparison.ipynb`; the confusion matrix (`reports/confusion_matrix.png`) shows near-perfect diagonal dominance with only a handful of off-diagonal misclassifications.

### 7.5 Extraction performance

Per-field extraction coverage is measured in `notebooks/exploration.ipynb` (Section 7, "Invoice Extraction Benchmark"), which runs the regex extractor across all invoices in the test set and reports the fraction of non-null values per field. Hand-written and real-world sample coverage is additionally validated by 38 unit tests in `tests/test_extraction.py`, all passing.

---

## 8. Testing and Quality Assurance

45 unit tests across two files, all passing:

tests/test_classifier.py — 15 tests
TestModelLoading (2) — model artefacts load, confidence dict has 4 keys
TestClassification (4) — one per class, clean samples
TestPredictPipeline (3) — extraction triggered only for invoices, empty input handled
TestRealDocuments (6) — real documents sourced from the web

tests/test_extraction.py — 30 tests
TestInvoiceNumber (5)
TestDates (7)
TestNamesAndAmounts (8)
TestHelpers (7) — normalisation unit tests
TestEdgeCases (3) — empty input, random text, schema completeness


Run with:

```bash
python3 demo/app.py # opens http://localhost:7860


Features:

- Text paste or file upload (`.txt`, `.pdf`, `.png`, `.jpg`, `.tiff`).
- Colour-coded classification label with an `OCR` badge when Tesseract was invoked.
- Per-class confidence bar chart (rescaled decision-function scores).
- Extracted-field table (invoices only) with value/Status columns.
- Five built-in example documents (one per class plus a freelancer invoice variant) for one-click demos.

---

## 10. Limitations and Future Work

### Known limitations

- **Invoice class is predominantly synthetic in training (~82%).** Real public invoice corpora on HuggingFace are frequently unavailable at load time, so the trained model has seen fewer real invoice formats than ideal. The extractor, however, is extensively tested against real online invoice samples.
- **Extractor is English-only at the keyword level.** Labels such as `"Factura"` (Spanish) or `"Rechnung"` (German) are partially recognised (via the `FACTURA-` prefix pattern) but issuer/recipient/total keywords are English. Multilingual coverage is an obvious extension.
- **Currency coverage.** The amount normaliser recognises `£$€`, `USD`, `EUR`, `GBP`. Other currencies (`JPY`, `CNY`, `INR`, `CHF`, ...) pass through unnormalised.
- **OCR quality on low-resolution scans.** The pre-processing pipeline is deliberately lightweight (no OpenCV) and will degrade on documents below ~200 DPI or with heavy skew. A more aggressive deskew + binarisation stage would improve recall at the cost of a larger dependency footprint.
- **Invoice layouts with line-item tables.** The extractor targets header-level fields; it does not attempt line-item parsing.

### Future work

1. Add FastAPI REST endpoint (scaffolding exists on the `ui-redesign` branch).
2. Multilingual extractor patterns (Spanish, French, German).
3. Confidence-thresholded abstention: return `unknown` when the top decision-function score is below a tuned threshold.
4. Active-learning loop: flag low-confidence invoices for human review and feed corrections back into the training set.

---

## 11. Reproducibility

### Environment

- Python ≥ 3.9
- Dependencies pinned via minimum versions in `requirements.txt`
- External binaries (required only for OCR): `tesseract`, `poppler`

### Deterministic seeds

- Dataset generation: `random.seed(42)` in `src/build_dataset.py`
- Classifier: `random_state=42` in `LinearSVC`
- 5-fold CV: `random_state=42` in `StratifiedKFold`

### End-to-end reproduction

```bash

pip install -r requirements.txt
python3 src/build_dataset.py # regenerate train.csv / test.csv (requires network)
python3 src/train.py # retrain, save artefacts to models/
python3 -m pytest tests/ -v # 45 tests must pass
jupyter notebook notebooks/model_comparison.ipynb # reproduce benchmark
python3 demo/app.py # launch UI


---

## 12. Appendix — Repository Layout

InformationExtraction/
├── README.md # quick-start guide
├── requirements.txt # Python dependencies
├── docs/
│ └── TECHNICAL_REPORT.md # this document
├── src/
│ ├── build_dataset.py # real-data download + synthetic generation
│ ├── train.py # feature engineering + LinearSVC training
│ ├── extract.py # regex-based invoice field extractor
│ └── predict.py # end-to-end inference (text/PDF/image)
├── demo/
│ └── app.py # Gradio web interface
├── notebooks/
│ ├── exploration.ipynb # EDA + top features + error analysis + extraction benchmark
│ └── model_comparison.ipynb # 5-model benchmark (CV + held-out)
├── tests/
│ ├── test_classifier.py # 15 classifier tests
│ └── test_extraction.py # 30 extractor tests
├── data/
│ ├── raw/ # auto-populated by build_dataset.py
│ └── processed/
│ ├── train.csv # 19,001 rows
│ └── test.csv # 3,125 rows
├── models/
│ ├── word_vectorizer.joblib
│ ├── char_vectorizer.joblib
│ ├── classifier.joblib
│ └── meta.json
└── reports/
├── cv_boxplot.png # 5-fold CV accuracy distribution
├── test_accuracy.png # test accuracy ranking
└── confusion_matrix.png # winner confusion matrix


---

*End of Technical Report.*