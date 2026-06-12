import os
import joblib
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score
from collector import collect_schedules, collect_daily_demand
from processor import process_demand_features, process_noshow_features
from config import MODEL_DIR, MIN_DATA_DAYS
from logger import logger

DEMAND_FEATURES = [
    "day_of_week", "week_of_month", "month",
    "is_lunch", "meal_type_enc", "rolling_7d", "rolling_14d"
]

NOSHOW_FEATURES = [
    "day_of_week", "is_lunch", "days_in_advance",
    "user_noshow_rate", "user_noshow_count"
]

def train_demand_model():
    df = collect_daily_demand()
    if df.empty or len(df) < MIN_DATA_DAYS:
        logger.warning(f"Dados insuficientes para treinar demanda.")
        return None
    df = process_demand_features(df)
    X = df[DEMAND_FEATURES].fillna(0)
    y = df["total_agendados"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=2)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    logger.info(f"Modelo de demanda treinado — MAE: {mae:.2f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/demand_model.joblib")
    joblib.dump(list(DEMAND_FEATURES), f"{MODEL_DIR}/demand_features.joblib")
    return model

def train_noshow_model():
    df = collect_schedules()
    if df.empty or len(df) < 50:
        logger.warning("Dados insuficientes para treinar no-show.")
        return None
    df = process_noshow_features(df)
    X = df[NOSHOW_FEATURES].fillna(0)
    y = df["is_noshow"]
    if y.nunique() < 2:
        logger.warning("Não há exemplos de no-show suficientes.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=2, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    logger.info(f"Modelo de no-show treinado — F1: {f1:.3f} | AUC: {auc:.3f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/noshow_model.joblib")
    joblib.dump(list(NOSHOW_FEATURES), f"{MODEL_DIR}/noshow_features.joblib")
    return model

def train_all():
    logger.info("Iniciando treino de todos os modelos...")
    train_demand_model()
    train_noshow_model()
    logger.info("Treino concluído.")
