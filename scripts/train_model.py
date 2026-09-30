"""Train the audit-risk classifier and save it with its metadata.

Run from anywhere:  python scripts/train_model.py
Outputs: app/model.joblib, app/model_meta.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "audit_dataset_updated.csv"
OUT_MODEL = ROOT / "app" / "model.joblib"
OUT_META = ROOT / "app" / "model_meta.json"

SEED = 42
MIN_DEPT_ROWS = 20  # drop departments too rare to learn from (e.g. single-row typos)
CATEGORICAL = ["Department", "Audit_Finding"]
NUMERIC = ["Month"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["Audit_Date"])
    df = df.dropna(subset=CATEGORICAL + ["Risk_Level", "Audit_Date"])
    df["Month"] = df["Audit_Date"].dt.month
    counts = df["Department"].value_counts()
    return df[df["Department"].isin(counts[counts >= MIN_DEPT_ROWS].index)].copy()


def make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def main() -> None:
    df = load()
    X, y = df[CATEGORICAL + NUMERIC], df["Risk_Level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    candidates = {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=SEED, n_jobs=-1
        ),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, est in candidates.items():
        s = cross_val_score(make_pipeline(est), X_train, y_train, cv=cv, scoring="f1_macro")
        cv_scores[name] = float(s.mean())
        print(f"{name:26s} CV F1-macro: {s.mean():.4f} ± {s.std():.4f}")

    best_name = max(
        (n for n in cv_scores if not n.startswith("Baseline")), key=cv_scores.get
    )
    best = make_pipeline(candidates[best_name]).fit(X_train, y_train)
    pred = best.predict(X_test)
    test_f1 = float(f1_score(y_test, pred, average="macro"))
    print(f"\nBest: {best_name} | held-out test F1-macro: {test_f1:.4f}")
    print(classification_report(y_test, pred, zero_division=0))

    # Refit on all data for deployment; metrics above come from the held-out split.
    final = make_pipeline(candidates[best_name]).fit(X, y)
    joblib.dump(final, OUT_MODEL)

    meta = {
        "model": best_name,
        "classes": list(final.classes_),
        "departments": sorted(df["Department"].unique()),
        "findings": sorted(df["Audit_Finding"].unique()),
        "cv_f1_macro": cv_scores,
        "test_f1_macro": test_f1,
        "baseline_f1_macro": cv_scores["Baseline (most frequent)"],
        "n_rows": int(len(df)),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Saved {OUT_MODEL.name} and {OUT_META.name}")


if __name__ == "__main__":
    main()
