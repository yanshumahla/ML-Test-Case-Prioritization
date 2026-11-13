from pathlib import Path
import io
import json
import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "priority_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "sample_test_cases.csv"
FEATURES = ["change_impact", "failure_rate", "severity", "execution_time",
            "recent_failures", "code_churn", "dependency_count", "last_result", "module"]
REQUIRED = ["test_case_id", "test_name", "module", "change_impact", "failure_rate",
            "severity", "execution_time", "recent_failures", "code_churn",
            "dependency_count", "last_result"]
PRIORITY_WEIGHT = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

app = FastAPI(title="ML Test Case Prioritization", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run: python ml/train_model.py")
    return joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics unavailable. Train the model first.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

@app.get("/api/sample")
def sample():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample data not generated.")
    return FileResponse(DATA_PATH, media_type="text/csv", filename="sample_test_cases.csv")

def validate_and_prepare(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")
    df = df.copy()
    numeric = ["change_impact", "failure_rate", "severity", "execution_time",
               "recent_failures", "code_churn", "dependency_count"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric].isna().any().any():
        raise HTTPException(status_code=400, detail="Numeric fields contain blank or non-numeric values.")
    if not df["change_impact"].between(1, 5).all() or not df["severity"].between(1, 5).all():
        raise HTTPException(status_code=400, detail="change_impact and severity must be between 1 and 5.")
    if not df["failure_rate"].between(0, 1).all():
        raise HTTPException(status_code=400, detail="failure_rate must be between 0 and 1.")
    if (df[["execution_time", "recent_failures", "code_churn", "dependency_count"]] < 0).any().any():
        raise HTTPException(status_code=400, detail="Execution time and count fields cannot be negative.")
    df["last_result"] = df["last_result"].astype(str).str.strip().str.title()
    if not df["last_result"].isin(["Pass", "Fail"]).all():
        raise HTTPException(status_code=400, detail="last_result must be Pass or Fail.")
    return df

@app.post("/api/prioritize")
async def prioritize(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV contains no rows.")
    if len(df) > 20000:
        raise HTTPException(status_code=400, detail="Maximum upload size is 20,000 test cases.")
    df = validate_and_prepare(df)
    bundle = load_model()
    model = bundle["pipeline"]
    X = df[FEATURES]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    df["predicted_priority"] = predictions
    df["risk_score"] = [
        round(sum(float(p) * PRIORITY_WEIGHT.get(c, 1) for c, p in zip(classes, row)) / 4 * 100, 1)
        for row in probabilities
    ]
    df["confidence"] = [round(float(max(row)) * 100, 1) for row in probabilities]
    df["priority_order"] = df["predicted_priority"].map(PRIORITY_WEIGHT)
    df = df.sort_values(["priority_order", "risk_score"], ascending=[False, False])
    df["rank"] = range(1, len(df) + 1)
    results = df.drop(columns=["priority_order"]).to_dict(orient="records")
    counts = df["predicted_priority"].value_counts().to_dict()
    return {"total": len(results), "priority_counts": counts, "results": results}

@app.post("/api/export")
async def export_results(payload: dict):
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="No results to export.")
    output = io.StringIO()
    pd.DataFrame(rows).to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prioritized_test_cases.csv"})
