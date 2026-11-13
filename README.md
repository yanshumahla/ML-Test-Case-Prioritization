# ML-Based Test Case Prioritization for Regression Testing

An advanced college project that ranks regression test cases using a supervised ML model.

## What it does
- Accepts test-case CSV files through a browser UI or API.
- Trains a Random Forest classifier using labeled historical/custom test data.
- Predicts a priority class (`Critical`, `High`, `Medium`, `Low`) and a risk score.
- Ranks test cases by predicted risk, then displays and exports the result.
- Reports model metrics on a held-out test split.

> Academic/demo note: `data/sample_test_cases.csv` is a reproducible synthetic dataset created for this project. It is not production incident data and must not be represented as a real public benchmark. The app also accepts an external/public dataset after mapping its columns to the schema below.

## Stack
Python, FastAPI, scikit-learn, pandas, HTML, CSS, JavaScript, CSV.

## Quick start (Windows PowerShell)
```powershell
cd ML_Test_Case_Prioritization
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python ml/train_model.py
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

## Quick start (macOS/Linux)
```bash
cd ML_Test_Case_Prioritization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ml/train_model.py
uvicorn app.main:app --reload
```

## CSV schema
Required input columns:
- `test_case_id`: unique identifier
- `test_name`: short test description
- `module`: application module
- `change_impact`: 1-5, estimated impact of recent code changes
- `failure_rate`: historical failure rate, 0-1
- `severity`: 1-5, business/technical severity if this test fails
- `execution_time`: historical execution time in seconds
- `recent_failures`: number of failures in recent runs
- `code_churn`: number of changed lines/files proxy (non-negative numeric)
- `dependency_count`: number of dependent components
- `last_result`: `Pass` or `Fail`
- `label`: historical priority label (`Critical`, `High`, `Medium`, `Low`) — required for training, optional for prediction uploads

Training data uses all columns except IDs/names/module and `label` as model features. Prediction uploads do not need `label`.

## API
- `GET /api/health` — service and model status
- `GET /api/metrics` — held-out model metrics
- `POST /api/prioritize` — upload a CSV and receive ranked results
- `GET /api/sample` — download sample data

## Methodology
1. Validate and clean the structured test-case data.
2. Train/test split with stratification.
3. Train a Random Forest classifier.
4. Evaluate with accuracy, macro precision/recall/F1 and confusion matrix.
5. Estimate a risk score from predicted class probabilities, then rank descending.

Priority labels are ordinal for display, but the model is a multiclass classifier. The score is a transparent weighted probability score, not a calibrated probability of a production defect.

## Important limitations / future improvements
- Synthetic labels are rule-generated, so metrics mainly verify pipeline correctness; they do not prove real-world effectiveness.
- Replace synthetic training data with a genuinely labeled historical dataset before making operational decisions.
- Add time-aware validation, cost-aware APFD evaluation, feature importance, authentication, audit logging, and integration with Jira/pytest/CI.
- Test execution time is included as a feature but ranking is not yet optimized for a fixed time budget.

## Suggested college report chapters
1. Introduction and problem statement
2. Literature review: regression testing and test case prioritization
3. Requirements and system design
4. Dataset and feature engineering
5. ML methodology and evaluation
6. Implementation and screenshots
7. Results, limitations and future scope
8. Conclusion
