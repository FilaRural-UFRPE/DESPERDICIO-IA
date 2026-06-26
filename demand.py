import os
import joblib
import pandas as pd
from datetime import date, timedelta
from fastapi import APIRouter
from pydantic import BaseModel
from config import MODEL_DIR
from logger import logger
from collector import collect_daily_demand
from processor import process_demand_features, MEAL_TYPE_ENC

router = APIRouter()

class DemandRequest(BaseModel):
    date: str
    meal_type: str = "lunch"
    meal_option: str = "essencial"

def _load_model():
    path = f"{MODEL_DIR}/demand_model.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@router.post("/predict")
def predict_demand(req: DemandRequest):
    model = _load_model()

    if model is None:
        df = collect_daily_demand()
        if df.empty:
            return {"predicted_meals": 0, "confidence": 0.0, "method": "no_data"}
        avg = int(df[df["schedule_type"] == req.meal_type]["total_agendados"].mean())
        return {"predicted_meals": avg, "confidence": 0.5, "method": "historical_mean"}

    target_date = pd.to_datetime(req.date)
    df = collect_daily_demand()
    df = process_demand_features(df)

    meal_type_enc = MEAL_TYPE_ENC.get(req.meal_option, 2)

    row = {
        "day_of_week": target_date.dayofweek,
        "week_of_month": target_date.day // 7,
        "month": target_date.month,
        "is_lunch": 1 if req.meal_type == "lunch" else 0,
        "meal_type_enc": meal_type_enc,
        "rolling_7d": df["total_agendados"].tail(7).mean() if not df.empty else 0,
        "rolling_14d": df["total_agendados"].tail(14).mean() if not df.empty else 0,
    }

    X = pd.DataFrame([row])
    pred = int(model.predict(X)[0])
    return {"predicted_meals": max(0, pred), "confidence": 0.85, "method": "xgboost"}

@router.get("/forecast")
def forecast(days: int = 7):
    results = []
    for i in range(1, days + 1):
        target = date.today() + timedelta(days=i)
        lunch = predict_demand(DemandRequest(date=str(target), meal_type="lunch"))
        dinner = predict_demand(DemandRequest(date=str(target), meal_type="dinner"))
        results.append({
            "date": str(target),
            "lunch": lunch["predicted_meals"],
            "dinner": dinner["predicted_meals"],
        })
    return results
