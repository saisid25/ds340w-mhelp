"""
Enhanced M-Help help-seeking classification experiment.

DS 340W project:
AI Detection of Mental Health Help-Seeking Signals from Social Media

This script keeps the project scope intentionally simple:
- input: one Reddit post at a time
- target: binary help-seeking label, help
- no user-level, multi-post, temporal, or longitudinal modeling

The enhancement over the first notebook is a comparison of multiple
classification techniques, including a character n-gram TF-IDF model.
Character n-grams are the main novelty block because they can capture
informal spelling, punctuation, and short emotional expressions common
in Reddit posts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def clean_text(text: str) -> str:
    """Light cleaning that preserves most wording from the original post."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(filename: str) -> pd.DataFrame:
    """Load one split and keep only the columns used in this project."""
    df = pd.read_csv(DATA_DIR / filename)
    df = df[["Text", "help"]].dropna().copy()
    df["help"] = df["help"].astype(int)
    df["clean_text"] = df["Text"].apply(clean_text)
    return df


def build_models() -> dict[str, Pipeline]:
    """Create baseline and enhanced models for comparison."""
    return {
        "Majority Class Baseline": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=1)),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Original: Word TF-IDF + Logistic Regression": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                ),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Balanced Logistic Regression": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=20000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=0.3,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Word TF-IDF + Linear SVM": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=20000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Complement Naive Bayes": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2),
                ),
                ("clf", ComplementNB()),
            ]
        ),
        "Novelty: Character TF-IDF + Linear SVM": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        max_features=20000,
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def majority_predictions(y_train: pd.Series, n_rows: int) -> list[int]:
    """Return the most frequent training label for every row."""
    majority_label = int(y_train.value_counts().idxmax())
    return [majority_label] * n_rows


def metric_row(model_name: str, split: str, y_true, y_pred) -> dict[str, float | str]:
    precision_help, recall_help, f1_help, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    return {
        "Model": model_name,
        "Split": split,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
        "Help Precision": precision_help[0],
        "Help Recall": recall_help[0],
        "Help F1": f1_help[0],
    }


def evaluate_model(model_name: str, model, train, val, test) -> list[dict[str, float | str]]:
    x_train = train["clean_text"]
    y_train = train["help"]
    rows = []

    if model_name == "Majority Class Baseline":
        for split_name, split_df in [("Validation", val), ("Test", test)]:
            y_pred = majority_predictions(y_train, len(split_df))
            rows.append(metric_row(model_name, split_name, split_df["help"], y_pred))
        return rows

    model.fit(x_train, y_train)
    for split_name, split_df in [("Validation", val), ("Test", test)]:
        y_pred = model.predict(split_df["clean_text"])
        rows.append(metric_row(model_name, split_name, split_df["help"], y_pred))
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    train = load_split("train_data.csv")
    val = load_split("val_data.csv")
    test = load_split("test_data.csv")

    print("Dataset sizes")
    print(f"Train: {len(train)} rows")
    print(f"Validation: {len(val)} rows")
    print(f"Test: {len(test)} rows")
    print()

    print("Label distributions")
    for split_name, split_df in [("Train", train), ("Validation", val), ("Test", test)]:
        print(split_name)
        print(split_df["help"].value_counts().sort_index().to_string())
    print()

    rows = []
    models = build_models()
    for model_name, model in models.items():
        rows.extend(evaluate_model(model_name, model, train, val, test))

    results = pd.DataFrame(rows)
    metric_cols = [
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Help Precision",
        "Help Recall",
        "Help F1",
    ]
    results[metric_cols] = results[metric_cols].round(3)
    results.to_csv(RESULTS_DIR / "comparison_results.csv", index=False)

    print("Comparison results")
    print(results.to_string(index=False))
    print()

    best_name = "Novelty: Character TF-IDF + Linear SVM"
    best_model = models[best_name]
    best_model.fit(train["clean_text"], train["help"])
    test_pred = best_model.predict(test["clean_text"])

    report_text = classification_report(
        test["help"],
        test_pred,
        target_names=["not help", "help"],
        zero_division=0,
    )
    matrix = confusion_matrix(test["help"], test_pred)

    (RESULTS_DIR / "best_model_test_classification_report.txt").write_text(report_text)
    pd.DataFrame(
        matrix,
        index=["actual_not_help", "actual_help"],
        columns=["predicted_not_help", "predicted_help"],
    ).to_csv(RESULTS_DIR / "best_model_test_confusion_matrix.csv")

    print(f"Best enhanced model selected for discussion: {best_name}")
    print("Test classification report")
    print(report_text)
    print("Test confusion matrix")
    print(matrix)


if __name__ == "__main__":
    main()
