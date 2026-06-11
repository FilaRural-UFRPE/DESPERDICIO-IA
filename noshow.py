import os
import joblib
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.config import MODEL_DIR
from app.pipeline.collector import collect_schedules
from app.pipeline.processor import process_noshow_features

router = APIRouter()

class NoshowRequest(BaseModel):
    user_cpf: str
    schedule_date: str
    meal_type: str = "lunch"

def _load_model():
    path = f"{MODEL_DIR}/noshow_model.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@router.post("/predict")
def predict_noshow(req: NoshowRequest):
    model = _load_model()

    df = collect_schedules()
    user_data = df[df["user_cpf"] == req.user_cpf] if not df.empty else pd.DataFrame()
    user_noshow_rate = user_data["is_noshow"].mean() if not user_data.empty else 0.2
    user_noshow_count = user_data["is_noshow"].sum() if not user_data.empty else 0

    if model is None:
        risk = "high" if user_noshow_rate > 0.4 else "medium" if user_noshow_rate > 0.2 else "low"
        return {"noshow_probability": round(user_noshow_rate, 2), "risk": risk, "method": "historical_rate"}

    target_date = pd.to_datetime(req.schedule_date)
    row = {
        "day_of_week": target_date.dayofweek,
        "is_lunch": 1 if req.meal_type == "lunch" else 0,
        "days_in_advance": 1,
        "user_noshow_rate": user_noshow_rate,
        "user_noshow_count": user_noshow_count,
    }
    X = pd.DataFrame([row])
    prob = float(model.predict_proba(X)[0][1])
    risk = "high" if prob > 0.6 else "medium" if prob > 0.3 else "low"
    return {"noshow_probability": round(prob, 2), "risk": risk, "method": "xgboost"}

@router.get("/summary")
def noshow_summary(date: str):
    df = collect_schedules()
    if df.empty:
        return {"high_risk_count": 0, "expected_attendance": 0}
    day_data = df[df["schedule_date"].astype(str).str.startswith(date)]
    total = len(day_data)
    avg_noshow_rate = day_data["is_noshow"].mean() if not day_data.empty else 0.2
    expected = int(total * (1 - avg_noshow_rate))
    high_risk = int(total * avg_noshow_rate)
    return {"high_risk_count": high_risk, "expected_attendance": expected, "total_scheduled": total}
