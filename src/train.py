"""
train.py
========
Trains a document classifier using TF-IDF + LinearSVC.

Pipeline
--------
  1. Load train.csv / test.csv
  2. TF-IDF vectorisation (char n-grams + word n-grams fused)
  3. LinearSVC with class-weight balancing
  4. Evaluate on test set → classification report + confusion matrix
  5. Persist model artefacts to models/

Usage
-----
    python3 src/train.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack

ROOT       = Path(__file__).resolve().parent.parent
DATA_PROC  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["invoice", "email", "scientific_report", "letter"]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data…")
    train = pd.read_csv(DATA_PROC / "train.csv").dropna(subset=["text", "label"])
    test  = pd.read_csv(DATA_PROC / "test.csv").dropna(subset=["text", "label"])

    train = train[train["label"].isin(LABELS)]
    test  = test[test["label"].isin(LABELS)]

    print(f"  Train: {len(train):,} | Test: {len(test):,}")
    print(f"  Train distribution: {dict(Counter(train['label']))}")
    print(f"  Test  distribution: {dict(Counter(test['label']))}")

    return train["text"].tolist(), train["label"].tolist(), \
           test["text"].tolist(),  test["label"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  — dual TF-IDF (word + char n-grams)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(X_train, X_test):
    """
    Two complementary vectorisers:
      word_tfidf : word unigrams + bigrams  (captures vocabulary patterns)
      char_tfidf : char 3–5-grams          (captures format/punctuation cues)
    """
    print("Building TF-IDF features…")

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=80_000,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        max_features=60_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X_train_word = word_vec.fit_transform(X_train)
    X_test_word  = word_vec.transform(X_test)

    X_train_char = char_vec.fit_transform(X_train)
    X_test_char  = char_vec.transform(X_test)

    X_train_feat = hstack([X_train_word, X_train_char])
    X_test_feat  = hstack([X_test_word,  X_test_char])

    print(f"  Feature matrix: {X_train_feat.shape[0]:,} × {X_train_feat.shape[1]:,}")
    return X_train_feat, X_test_feat, word_vec, char_vec


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(y_train):
    """
    LinearSVC is fast, memory-efficient, and excels at high-dimensional sparse
    text features.  class_weight='balanced' handles any remaining imbalance.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))

    clf = LinearSVC(
        C=1.0,
        class_weight=cw_dict,
        max_iter=2000,
        random_state=42,
    )
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate():
    X_train, y_train, X_test, y_test = load_data()

    X_train_feat, X_test_feat, word_vec, char_vec = build_features(X_train, X_test)

    print("\nTraining LinearSVC…")
    clf = build_model(y_train)
    clf.fit(X_train_feat, y_train)

    print("Evaluating…")
    y_pred = clf.predict(X_test_feat)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'═'*55}")
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print(f"{'═'*55}")
    print(classification_report(y_test, y_pred, target_names=sorted(set(y_test))))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':20s}" + "  ".join(f"{l[:6]:>8}" for l in LABELS))
    for i, row_label in enumerate(LABELS):
        print(f"  {row_label:<20s}" + "  ".join(f"{v:>8}" for v in cm[i]))

    # ── persist ────────────────────────────────────────────────────────────
    print("\nSaving model artefacts…")
    joblib.dump(word_vec, MODELS_DIR / "word_vectorizer.joblib")
    joblib.dump(char_vec, MODELS_DIR / "char_vectorizer.joblib")
    joblib.dump(clf,      MODELS_DIR / "classifier.joblib")

    meta = {
        "labels":   LABELS,
        "accuracy": round(acc, 4),
        "model":    "LinearSVC",
        "features": "TF-IDF word(1-2)+char(3-5) n-grams",
    }
    with open(MODELS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Artefacts saved to {MODELS_DIR}/")
    print("  word_vectorizer.joblib")
    print("  char_vectorizer.joblib")
    print("  classifier.joblib")
    print("  meta.json")

    return acc


if __name__ == "__main__":
    acc = train_and_evaluate()
    sys.exit(0 if acc > 0.80 else 1)
