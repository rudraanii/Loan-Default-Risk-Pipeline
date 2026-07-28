# Loan Default Risk Prediction — End-to-End ML Pipeline

A complete, production-shaped ML pipeline for predicting loan default risk:
data generation → model training & comparison → REST API serving → Docker
deployment. Built to demonstrate the full lifecycle expected of an ML
engineer, not just a Jupyter notebook.

## Why this project

Recruiters see hundreds of "trained a model, got 90% accuracy" notebooks.
This repo instead shows:
- Comparing multiple model families and selecting by the right metric (ROC-AUC, not accuracy — the data is imbalanced)
- A real serving layer (FastAPI) with input validation
- Containerization for deployment (Docker)
- Saved metrics/model artifacts for reproducibility

## Architecture

```
data/generate_data.py  →  data/loan_data.csv
                                 │
                                 ▼
                     model/train.py
         (Logistic Regression / Random Forest / Gradient Boosting)
                                 │
                                 ▼
               model/loan_model.joblib + metrics.json
                                 │
                                 ▼
                     app/main.py (FastAPI)
                                 │
                                 ▼
                          Docker container
```

## Results (on synthetic held-out test set)

| Model               | ROC-AUC | Precision | Recall | F1   |
|---------------------|---------|-----------|--------|------|
| Logistic Regression | 0.912   | 0.28      | 0.92   | 0.43 |
| Random Forest        | 0.902   | 0.45      | 0.57   | 0.50 |
| Gradient Boosting     | 0.906   | 0.72      | 0.39   | 0.51 |

Logistic Regression was selected for deployment — highest ROC-AUC and best
recall on the minority (default) class, which matters more than raw accuracy
in credit risk since missing a defaulter is costlier than a false alarm.

> Dataset is synthetically generated (`data/generate_data.py`) so the repo
> runs fully offline. Swap in a real dataset (e.g. LendingClub, Home Credit
> Default Risk on Kaggle) by matching the same column schema.

## Setup

```bash
git clone https://github.com/<your-username>/loan-default-pipeline.git
cd loan-default-pipeline
pip install -r requirements.txt

python data/generate_data.py   # generate dataset
python model/train.py          # train + compare models
uvicorn app.main:app --reload  # serve API
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 34,
        "income": 48000,
        "loan_amount": 20000,
        "credit_score": 610,
        "employment_years": 2.5,
        "existing_debt": 12000,
        "num_credit_lines": 6
      }'
```

**Response**
```json
{
  "default_probability": 0.3821,
  "risk_level": "Medium",
  "prediction": "No Default"
}
```

## Run with Docker

```bash
docker build -t loan-default-api .
docker run -p 8000:8000 loan-default-api
```

## Project Structure

```
├── data/
│   ├── generate_data.py
│   └── loan_data.csv
├── model/
│   ├── train.py
│   ├── loan_model.joblib
│   └── metrics.json
├── app/
│   └── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Possible Extensions
- Add SHAP explainability endpoint for individual predictions
- Add MLflow experiment tracking
- CI/CD via GitHub Actions to retrain and redeploy on new data
- Deploy to AWS ECS / Azure Container Apps

## License
MIT
