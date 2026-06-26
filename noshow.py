import os
import joblib
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from config import MODEL_DIR
from collector import collect_schedules
from processor import PRIOR_NOSHOW_BY_WEEKDAY

router = APIRouter()

NOSHOW_FEATURES = [
    "day_of_week", "month", "is_lunch", "is_weekend", "is_monday",
    "days_in_advance", "meal_type_enc",
    "user_noshow_rate", "user_noshow_count", "user_total_schedules",
    "user_reliability_score", "prior_noshow",
]


class NoshowRequest(BaseModel):
    user_cpf: str
    schedule_date: str
    meal_type: str = "lunch"
    days_in_advance: int = 1


def _load_model():
    path = f"{MODEL_DIR}/noshow_model.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _prior_predict(target_date: pd.Timestamp, user_cpf: str, df: pd.DataFrame) -> dict:
    """
    Fallback inteligente para no-show.
    Combina prior por dia da semana com histórico do utilizador (se existir).
    """
    prior_rate = PRIOR_NOSHOW_BY_WEEKDAY.get(target_date.dayofweek, 0.2)

    if not df.empty:
        user_data = df[df["user_cpf"] == user_cpf]
        if not user_data.empty and len(user_data) >= 2:
            observed_rate = user_data["is_noshow"].mean()
            weight = min(len(user_data) / 5, 1.0)
            final_rate = weight * observed_rate + (1 - weight) * prior_rate
        else:
            final_rate = prior_rate
    else:
        final_rate = prior_rate

    risk = "alto" if final_rate > 0.35 else "médio" if final_rate > 0.2 else "baixo"
    return {
        "noshow_probability": round(final_rate, 3),
        "risk": risk,
        "method": "prior_with_user_history",
        "note": "Estimativa baseada em padrões típicos + histórico do utilizador.",
    }


@router.post("/predict")
def predict_noshow(req: NoshowRequest):
    model = _load_model()
    df    = collect_schedules()
    target_date = pd.to_datetime(req.schedule_date)

    # Dados do utilizador
    user_data = df[df["user_cpf"] == req.user_cpf] if not df.empty else pd.DataFrame()
    user_noshow_rate      = user_data["is_noshow"].mean()  if not user_data.empty else None
    user_noshow_count     = int(user_data["is_noshow"].sum())   if not user_data.empty else 0
    user_total_schedules  = len(user_data)

    # Blending com prior se utilizador tem poucos dados
    prior_rate = PRIOR_NOSHOW_BY_WEEKDAY.get(target_date.dayofweek, 0.2)
    if user_noshow_rate is None or user_total_schedules < 3:
        blend_weight     = min(user_total_schedules / 5, 1.0)
        user_noshow_rate = blend_weight * (user_noshow_rate or prior_rate) + (1 - blend_weight) * prior_rate

    if model is None:
        return _prior_predict(target_date, req.user_cpf, df)

    row = {
        "day_of_week":          target_date.dayofweek,
        "month":                target_date.month,
        "is_lunch":             1 if req.meal_type == "lunch" else 0,
        "is_weekend":           1 if target_date.dayofweek >= 5 else 0,
        "is_monday":            1 if target_date.dayofweek == 0 else 0,
        "days_in_advance":      req.days_in_advance,
        "meal_type_enc":        2,
        "user_noshow_rate":     user_noshow_rate,
        "user_noshow_count":    user_noshow_count,
        "user_total_schedules": user_total_schedules,
        "user_reliability_score": 1 - user_noshow_rate,
        "prior_noshow":         prior_rate,
    }

    X    = pd.DataFrame([row])[NOSHOW_FEATURES].fillna(0)
    prob = float(model.predict_proba(X)[0][1])

    # Suavização: blending com prior quando há poucos dados globais
    if len(df) < 30:
        weight = len(df) / 30
        prob   = weight * prob + (1 - weight) * prior_rate

    risk = "alto" if prob > 0.35 else "médio" if prob > 0.2 else "baixo"
    return {
        "noshow_probability": round(prob, 3),
        "risk": risk,
        "method": "xgboost",
    }


@router.get("/summary")
def noshow_summary(date: str):
    """
    Resumo de no-show para uma data.
    Funciona com qualquer quantidade de dados.
    """
    df = collect_schedules()
    target_date = pd.to_datetime(date)
    prior_rate  = PRIOR_NOSHOW_BY_WEEKDAY.get(target_date.dayofweek, 0.2)

    if df.empty:
        # Sem dados: usa prior puro
        prior_total = 50  # estimativa base
        return {
            "high_risk_count":     int(prior_total * prior_rate),
            "expected_attendance": int(prior_total * (1 - prior_rate)),
            "total_scheduled":     prior_total,
            "noshow_rate":         prior_rate,
            "method":              "prior_no_data",
            "note":                "Sem agendamentos reais. Estimativa baseada em padrões típicos.",
        }

    # Tenta encontrar agendamentos do dia exato
    df["schedule_date_str"] = df["schedule_date"].astype(str).str[:10]
    day_data = df[df["schedule_date_str"] == date[:10]]

    if day_data.empty:
        # Sem dados para esse dia: usa prior + média global
        global_rate = df["is_noshow"].mean() if not df.empty else prior_rate
        weight      = min(len(df) / 50, 1.0)
        final_rate  = weight * global_rate + (1 - weight) * prior_rate
        total       = 50  # estimativa
        return {
            "high_risk_count":     int(total * final_rate),
            "expected_attendance": int(total * (1 - final_rate)),
            "total_scheduled":     0,
            "noshow_rate":         round(final_rate, 3),
            "method":              "prior_with_global_history",
            "note":                "Sem agendamentos para esse dia. Estimativa baseada no histórico geral.",
        }

    total      = len(day_data)
    obs_rate   = day_data["is_noshow"].mean()
    weight     = min(total / 20, 1.0)
    final_rate = weight * obs_rate + (1 - weight) * prior_rate

    return {
        "high_risk_count":     int(total * final_rate),
        "expected_attendance": int(total * (1 - final_rate)),
        "total_scheduled":     total,
        "noshow_rate":         round(final_rate, 3),
        "method":              "observed_with_prior_blending",
    }
