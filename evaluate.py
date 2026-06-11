"""Script para avaliar os modelos treinados."""
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from app.pipeline.collector import collect_schedules, collect_daily_demand
from app.pipeline.processor import process_demand_features, process_noshow_features
from app.utils.config import MODEL_DIR

def evaluate_demand():
    df = collect_daily_demand()
    df = process_demand_features(df)
    features = joblib.load(f"{MODEL_DIR}/demand_features.joblib")
    model = joblib.load(f"{MODEL_DIR}/demand_model.joblib")
    X = df[features].fillna(0)
    y = df["total_agendados"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Demanda — MAE: {mae:.2f}")

def evaluate_noshow():
    df = collect_schedules()
    df = process_noshow_features(df)
    features = joblib.load(f"{MODEL_DIR}/noshow_features.joblib")
    model = joblib.load(f"{MODEL_DIR}/noshow_model.joblib")
    X = df[features].fillna(0)
    y = df["is_noshow"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"No-Show — F1: {f1_score(y_test, preds):.3f} | AUC: {roc_auc_score(y_test, probs):.3f}")

if __name__ == "__main__":
    evaluate_demand()
    evaluate_noshow()
