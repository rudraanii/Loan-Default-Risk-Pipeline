"""
Loan Default Risk API — serves the trained classifier via REST.
Run: uvicorn app.main:app --reload
"""
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Loan Default Risk API", version="1.0.0")

MODEL_PATH = "model/loan_model.joblib"
METRICS_PATH = "model/metrics.json"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

with open(METRICS_PATH) as f:
    metrics = json.load(f)


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: float = Field(..., ge=300, le=850)
    employment_years: float = Field(..., ge=0)
    existing_debt: float = Field(..., ge=0)
    num_credit_lines: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    default_probability: float
    risk_level: str
    prediction: str


def risk_bucket(prob: float) -> str:
    if prob < 0.15:
        return "Low"
    if prob < 0.4:
        return "Medium"
    return "High"


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    try:
        X = np.array([[getattr(application, f) for f in features]])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return PredictionResponse(
        default_probability=round(float(prob), 4),
        risk_level=risk_bucket(prob),
        prediction="Default" if prob >= 0.5 else "No Default",
    )


@app.get("/model-info")
async def model_info():
    return metrics


@app.get("/health")
async def health():
    return {"status": "ok"}
