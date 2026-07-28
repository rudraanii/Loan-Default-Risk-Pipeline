"""
Trains and evaluates a loan-default classifier.
Compares Logistic Regression vs Random Forest vs Gradient Boosting,
picks the best by ROC-AUC on a held-out test set, and saves the model.

Run: python model/train.py
"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report

DATA_PATH = "data/loan_data.csv"
MODEL_PATH = "model/loan_model.joblib"
METRICS_PATH = "model/metrics.json"

FEATURES = [
    "age", "income", "loan_amount", "credit_score",
    "employment_years", "existing_debt", "num_credit_lines",
]
TARGET = "default"


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }

    results = {}
    best_name, best_model, best_auc = None, None, -1

    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        preds = model.predict(X_test_s)

        auc = roc_auc_score(y_test, probs)
        results[name] = {
            "roc_auc": round(auc, 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=["No Default", "Default"]))
        print(f"ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")

    joblib.dump({"model": best_model, "scaler": scaler, "features": FEATURES}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"Saved model to {MODEL_PATH} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
