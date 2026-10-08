"""Train a classifier on the Breast Cancer dataset.

Hyperparameters live in params.json so every experiment is a small,
reviewable Git diff. Results are written to metrics.json and committed
alongside the code that produced them.
"""
import json
import subprocess

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(params):
    if params["model"] == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=params["C"], max_iter=1000),
        )
    if params["model"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
    raise ValueError(f"Unknown model: {params['model']}")


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def train(params):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )
    model = build_model(params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "model": params["model"],
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "trained_on_commit": git_commit_hash(),
    }


if __name__ == "__main__":
    with open("params.json") as f:
        params = json.load(f)
    metrics = train(params)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
