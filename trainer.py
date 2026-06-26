import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score
from collector import collect_schedules, collect_daily_demand
from processor import (
    process_demand_features, process_noshow_features,
    MEAL_TYPE_ENC, PRIOR_DEMAND_BY_WEEKDAY, PRIOR_NOSHOW_BY_WEEKDAY
)
from config import MODEL_DIR, MIN_DATA_DAYS
from logger import logger

DEMAND_FEATURES = [
    "day_of_week", "week_of_month", "month", "quarter",
    "is_lunch", "is_weekend", "is_monday", "is_friday",
    "meal_type_enc",
    "rolling_3d", "rolling_7d", "rolling_14d", "rolling_30d",
    "lag_1d", "lag_7d", "trend_7d", "std_7d",
    "prior_demand",
]

NOSHOW_FEATURES = [
    "day_of_week", "month", "is_lunch", "is_weekend", "is_monday",
    "days_in_advance", "meal_type_enc",
    "user_noshow_rate", "user_noshow_count", "user_total_schedules",
    "user_reliability_score", "prior_noshow",
]

# Mínimos para cada modelo — muito baixos para funcionar cedo
MIN_DEMAND_ROWS = 3
MIN_NOSHOW_ROWS = 10


def _safe_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    for f in features:
        if f not in df.columns:
            df[f] = 0
    return df[features].fillna(0)


def _prior_demand_model():
    """
    Modelo prior baseado em conhecimento de domínio.
    Usado quando não há dados suficientes para treinar o XGBoost.
    """
    return None  # Sinaliza que deve usar o prior diretamente


def train_demand_model():
    df = collect_daily_demand()

    if df.empty or len(df) < MIN_DEMAND_ROWS:
        logger.warning(f"Poucos dados para demanda ({len(df)} registos). "
                       f"A IA usará prior por dia da semana.")
        # Salva os priors para que o predict os use
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(None, f"{MODEL_DIR}/demand_model.joblib")
        joblib.dump({"method": "prior", "rows": len(df)}, f"{MODEL_DIR}/demand_metrics.joblib")
        return None

    df = process_demand_features(df)
    X = _safe_features(df, DEMAND_FEATURES)
    y = df["total_agendados"]

    # Com poucos dados: sem split de validação, treina em tudo
    if len(df) < 20:
        logger.info(f"Poucos dados ({len(df)}). Treinando sem validação.")
        model = XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=1.0, colsample_bytree=1.0,
            random_state=42, n_jobs=2,
        )
        model.fit(X, y)
        mae = mean_absolute_error(y, model.predict(X))
        logger.info(f"Modelo demanda (poucos dados) — MAE treino: {mae:.2f}")
        metrics = {"method": "xgboost_small", "mae": mae, "rows": len(df)}
    else:
        # Dados suficientes: split temporal
        split = max(1, int(len(df) * 0.8))
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=2,
        )
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        logger.info(f"Modelo demanda — MAE validação: {mae:.2f}")
        metrics = {"method": "xgboost", "mae": mae, "rows": len(df)}

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/demand_model.joblib")
    joblib.dump(DEMAND_FEATURES, f"{MODEL_DIR}/demand_features.joblib")
    joblib.dump(metrics, f"{MODEL_DIR}/demand_metrics.joblib")
    return model


def train_noshow_model():
    df = collect_schedules()

    if df.empty or len(df) < MIN_NOSHOW_ROWS:
        logger.warning(f"Poucos dados para no-show ({len(df)} registos). "
                       f"A IA usará prior por dia da semana.")
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(None, f"{MODEL_DIR}/noshow_model.joblib")
        joblib.dump({"method": "prior", "rows": len(df)}, f"{MODEL_DIR}/noshow_metrics.joblib")
        return None

    df = process_noshow_features(df)
    X = _safe_features(df, NOSHOW_FEATURES)
    y = df["is_noshow"]

    if y.nunique() < 2:
        logger.warning("Sem exemplos de no-show ainda. Usando prior.")
        joblib.dump(None, f"{MODEL_DIR}/noshow_model.joblib")
        return None

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    # Com poucos dados: sem estratificação, treina em tudo
    if len(df) < 50:
        logger.info(f"Poucos dados no-show ({len(df)}). Treinando sem split.")
        model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            scale_pos_weight=pos_weight,
            random_state=42, n_jobs=2, eval_metric="logloss",
        )
        model.fit(X, y)
        f1  = f1_score(y, model.predict(X), zero_division=0)
        auc = 0.0
        metrics = {"method": "xgboost_small", "f1": f1, "auc": auc, "rows": len(df)}
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42, n_jobs=2, eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        f1  = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        metrics = {"method": "xgboost", "f1": f1, "auc": auc, "rows": len(df)}

    logger.info(f"Modelo no-show — F1: {metrics['f1']:.3f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/noshow_model.joblib")
    joblib.dump(NOSHOW_FEATURES, f"{MODEL_DIR}/noshow_features.joblib")
    joblib.dump(metrics, f"{MODEL_DIR}/noshow_metrics.joblib")
    return model


def get_model_metrics() -> dict:
    metrics = {}
    for name in ["demand_metrics", "noshow_metrics"]:
        path = f"{MODEL_DIR}/{name}.joblib"
        if os.path.exists(path):
            metrics[name] = joblib.load(path)
    return metrics


def train_all():
    logger.info("═══ Iniciando treino de todos os modelos ═══")
    train_demand_model()
    train_noshow_model()
    logger.info("═══ Treino concluído ═══")
