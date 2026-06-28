import os
import joblib
import pandas as pd
import numpy as np
from datetime import date, timedelta
from fastapi import APIRouter
from pydantic import BaseModel
from config import MODEL_DIR
from logger import logger
from collector import collect_daily_demand
from processor import process_demand_features, MEAL_TYPE_ENC, PRIOR_DEMAND_BY_WEEKDAY

router = APIRouter()

DEMAND_FEATURES = [
    "day_of_week", "week_of_month", "month", "quarter",
    "is_lunch", "is_weekend", "is_monday", "is_friday",
    "meal_type_enc",
    "rolling_3d", "rolling_7d", "rolling_14d", "rolling_30d",
    "lag_1d", "lag_7d", "trend_7d", "std_7d",
    "prior_demand",
]

MEAL_OPTION_MULTIPLIER = {
    "select":      0.20,
    "leve_sabor":  0.30,
    "essencial":   0.45,
    "vegetariano": 0.05,
}

# Cache global para evitar chamadas repetidas à API
_df_cache = None
_df_cache_date = None


def _get_df():
    """Retorna DataFrame com cache por dia."""
    global _df_cache, _df_cache_date
    today = str(date.today())
    if _df_cache is None or _df_cache_date != today:
        _df_cache = collect_daily_demand()
        _df_cache_date = today
    return _df_cache


class DemandRequest(BaseModel):
    date: str
    meal_type: str = "lunch"
    meal_option: str = "essencial"


def _load_model():
    path = f"{MODEL_DIR}/demand_model.joblib"
    if not os.path.exists(path):
        return None
    m = joblib.load(path)
    # Se joblib salvou None (prior), retorna None
    return m if hasattr(m, 'predict') else None


def _prior_predict(target_date: pd.Timestamp, meal_type: str, meal_option: str, df: pd.DataFrame) -> dict:
    prior = PRIOR_DEMAND_BY_WEEKDAY.get(target_date.dayofweek, 100)

    if not df.empty:
        df2 = df.copy()
        df2["schedule_date"] = pd.to_datetime(df2["schedule_date"])
        same_day = df2[df2["schedule_date"].dt.dayofweek == target_date.dayofweek]
        if not same_day.empty:
            observed_mean = same_day["total_agendados"].mean()
            weight = min(len(same_day) / 10, 1.0)
            prior = weight * observed_mean + (1 - weight) * prior

    if meal_type == "dinner":
        prior *= 0.6

    option_share = MEAL_OPTION_MULTIPLIER.get(meal_option, 0.45)
    predicted = int(prior * option_share)

    return {
        "predicted_meals": max(0, predicted),
        "confidence": 0.45,
        "method": "prior_with_blending",
    }


def _build_row(target_date: pd.Timestamp, df: pd.DataFrame, meal_type: str, meal_option: str) -> dict:
    global_mean = df["total_agendados"].mean() if not df.empty else 100

    def tail_mean(n): return df["total_agendados"].tail(n).mean() if not df.empty else global_mean
    def tail_val(n):  return float(df["total_agendados"].iloc[-n]) if len(df) >= n else global_mean

    rolling_7d = tail_mean(7)
    prior = PRIOR_DEMAND_BY_WEEKDAY.get(target_date.dayofweek, 100)

    return {
        "day_of_week":   target_date.dayofweek,
        "week_of_month": target_date.day // 7,
        "month":         target_date.month,
        "quarter":       (target_date.month - 1) // 3 + 1,
        "is_lunch":      1 if meal_type == "lunch" else 0,
        "is_weekend":    1 if target_date.dayofweek >= 5 else 0,
        "is_monday":     1 if target_date.dayofweek == 0 else 0,
        "is_friday":     1 if target_date.dayofweek == 4 else 0,
        "meal_type_enc": MEAL_TYPE_ENC.get(meal_option, 2),
        "rolling_3d":    tail_mean(3),
        "rolling_7d":    rolling_7d,
        "rolling_14d":   tail_mean(14),
        "rolling_30d":   tail_mean(30),
        "lag_1d":        tail_val(1),
        "lag_7d":        tail_val(7),
        "trend_7d":      tail_val(1) - rolling_7d,
        "std_7d":        df["total_agendados"].tail(7).std() if not df.empty else 0,
        "prior_demand":  prior,
    }


@router.post("/predict")
def predict_demand(req: DemandRequest):
    model = _load_model()
    df    = _get_df()
    target_date = pd.to_datetime(req.date)

    if model is None:
        return _prior_predict(target_date, req.meal_type, req.meal_option, df)

    row = _build_row(target_date, df, req.meal_type, req.meal_option)
    X   = pd.DataFrame([row])
    for f in DEMAND_FEATURES:
        if f not in X.columns:
            X[f] = 0
    X = X[DEMAND_FEATURES].fillna(0)

    xgb_pred = float(model.predict(X)[0])

    # Blending: combina XGBoost com prior
    # Quanto mais dados, mais peso no XGBoost
    n_records = len(df) if not df.empty else 0
    xgb_weight = min(n_records / 100, 0.8)  # máximo 80% XGBoost até 100 registos

    prior_result = _prior_predict(target_date, req.meal_type, req.meal_option, df)
    prior_pred   = prior_result["predicted_meals"]

    blended = int(xgb_weight * xgb_pred + (1 - xgb_weight) * prior_pred)
    blended = max(0, blended)

    std = df["total_agendados"].tail(7).std() if not df.empty else blended * 0.2

    method = "xgboost" if xgb_weight >= 0.8 else f"blended_{int(xgb_weight*100)}pct_xgb"

    return {
        "predicted_meals": blended,
        "confidence_interval": {"lower": max(0, blended - int(std)), "upper": blended + int(std)},
        "confidence": 0.5 + xgb_weight * 0.35,
        "method": method,
        "meal_type": req.meal_type,
        "meal_option": req.meal_option,
    }


@router.get("/forecast")
def forecast(days: int = 7):
    """Previsão de demanda para os próximos N dias."""
    df = _get_df()  # carrega uma vez só
    results = []
    for i in range(1, days + 1):
        target = date.today() + timedelta(days=i)
        lunch  = predict_demand(DemandRequest(date=str(target), meal_type="lunch"))
        dinner = predict_demand(DemandRequest(date=str(target), meal_type="dinner"))
        results.append({
            "date":   str(target),
            "lunch":  lunch["predicted_meals"],
            "dinner": dinner["predicted_meals"],
            "method": lunch["method"],
        })
    return results


@router.get("/forecast/by-meal-option")
def forecast_by_meal_option(days: int = 7):
    """Previsão detalhada por tipo de refeição."""
    meal_options = list(MEAL_OPTION_MULTIPLIER.keys())
    results = []
    for i in range(1, days + 1):
        target = date.today() + timedelta(days=i)
        day_data = {"date": str(target), "by_option": {}}
        for option in meal_options:
            lunch  = predict_demand(DemandRequest(date=str(target), meal_type="lunch",  meal_option=option))
            dinner = predict_demand(DemandRequest(date=str(target), meal_type="dinner", meal_option=option))
            day_data["by_option"][option] = {
                "lunch":  lunch["predicted_meals"],
                "dinner": dinner["predicted_meals"],
            }
        results.append(day_data)
    return results


@router.get("/waste-risk")
def waste_risk(days: int = 7):
    """Estima risco de desperdício por dia."""
    from noshow import noshow_summary
    results = []
    for i in range(1, days + 1):
        target = str(date.today() + timedelta(days=i))
        lunch_pred  = predict_demand(DemandRequest(date=target, meal_type="lunch"))
        dinner_pred = predict_demand(DemandRequest(date=target, meal_type="dinner"))
        try:
            ns = noshow_summary(date=target)
            noshow_rate = 1 - (ns["expected_attendance"] / max(ns["total_scheduled"], 1))
        except Exception:
            noshow_rate = 0.2

        risk_level = "alto" if noshow_rate > 0.35 else "médio" if noshow_rate > 0.2 else "baixo"
        results.append({
            "date":        target,
            "noshow_rate": round(noshow_rate, 3),
            "risk_level":  risk_level,
            "lunch":  {
                "predicted":       lunch_pred["predicted_meals"],
                "estimated_waste": int(lunch_pred["predicted_meals"] * noshow_rate),
                "recommend_prep":  max(0, int(lunch_pred["predicted_meals"] * (1 - noshow_rate))),
            },
            "dinner": {
                "predicted":       dinner_pred["predicted_meals"],
                "estimated_waste": int(dinner_pred["predicted_meals"] * noshow_rate),
                "recommend_prep":  max(0, int(dinner_pred["predicted_meals"] * (1 - noshow_rate))),
            },
        })
    return results
