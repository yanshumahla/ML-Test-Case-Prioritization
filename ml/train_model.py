"""Train and evaluate the prioritization model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample_test_cases.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "priority_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

NUMERIC = ["change_impact", "failure_rate", "severity", "execution_time",
           "recent_failures", "code_churn", "dependency_count"]
CATEGORICAL = ["last_result", "module"]
FEATURES = NUMERIC + CATEGORICAL
CLASSES = ["Low", "Medium", "High", "Critical"]

def main():
    if not DATA.exists():
        from ml.generate_dataset import main as generate
        generate()
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42,
            min_samples_leaf=2, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "labels": CLASSES,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=CLASSES).tolist(),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
        "note": "Synthetic rule-derived demonstration data; metrics are not evidence of production performance."
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "features": FEATURES, "classes": CLASSES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(classification_report(y_test, pred, zero_division=0))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
