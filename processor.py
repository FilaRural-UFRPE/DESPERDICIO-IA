import pandas as pd
import numpy as np
from logger import logger

MEAL_TYPE_ENC = {
    "select":      0,
    "leve_sabor":  1,
    "essencial":   2,
    "vegetariano": 3,
}

MEAL_TYPE_LABELS = {v: k for k, v in MEAL_TYPE_ENC.items()}

DAY_NAMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

# Distribuição padrão por dia da semana (conhecimento de domínio)
# Usado quando não há dados suficientes
PRIOR_DEMAND_BY_WEEKDAY = {
    0: 120,  # Segunda
    1: 110,  # Terça
    2: 115,  # Quarta
    3: 108,  # Quinta
    4: 95,   # Sexta
    5: 40,   # Sábado
    6: 20,   # Domingo
}

PRIOR_NOSHOW_BY_WEEKDAY = {
    0: 0.25,  # Segunda — maior no-show (fim de semana prolongado)
    1: 0.18,
    2: 0.15,
    3: 0.17,
    4: 0.22,  # Sexta — mais faltas
    5: 0.30,
    6: 0.35,
}


def _safe_rolling(series: pd.Series, window: int, func="mean") -> pd.Series:
    """Rolling com min_periods=1 para funcionar com poucos dados."""
    r = series.rolling(window, min_periods=1)
    return r.mean() if func == "mean" else r.std().fillna(0)


def process_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["day_of_week"]   = df["schedule_date"].dt.dayofweek
    df["week_of_month"] = df["schedule_date"].dt.day // 7
    df["month"]         = df["schedule_date"].dt.month
    df["quarter"]       = df["schedule_date"].dt.quarter
    df["is_weekend"]    = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_monday"]     = (df["day_of_week"] == 0).astype(int)
    df["is_friday"]     = (df["day_of_week"] == 4).astype(int)
    meal_col = "meal_type" if "meal_type" in df.columns else "schedule_type"
    df["is_lunch"]      = (df[meal_col] == "lunch").astype(int)

    option_col = "meal_option" if "meal_option" in df.columns else "meal_type"
    df["meal_type_enc"] = df[option_col].map(MEAL_TYPE_ENC).fillna(2).astype(int)

    df = df.sort_values("schedule_date").reset_index(drop=True)

    n = len(df)

    # Rolling — funciona mesmo com 1 registo
    df["rolling_3d"]  = _safe_rolling(df["total_agendados"], min(3,  n))
    df["rolling_7d"]  = _safe_rolling(df["total_agendados"], min(7,  n))
    df["rolling_14d"] = _safe_rolling(df["total_agendados"], min(14, n))
    df["rolling_30d"] = _safe_rolling(df["total_agendados"], min(30, n))

    # Lag features — se não há dados anteriores, usa a média global
    global_mean = df["total_agendados"].mean()
    df["lag_1d"] = df["total_agendados"].shift(1).fillna(global_mean)
    df["lag_7d"] = df["total_agendados"].shift(min(7, n - 1)).fillna(global_mean)

    df["trend_7d"] = df["total_agendados"] - df["rolling_7d"]
    df["std_7d"]   = _safe_rolling(df["total_agendados"], min(7, n), func="std")

    # Prior por dia da semana (ancora a predição quando há poucos dados)
    df["prior_demand"] = df["day_of_week"].map(PRIOR_DEMAND_BY_WEEKDAY).fillna(100)

    logger.info(f"Features de demanda geradas: {n} registos.")
    return df


def process_noshow_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["created_at"]    = pd.to_datetime(df["created_at"])

    df["schedule_date"] = df["schedule_date"].dt.tz_localize(None) if df["schedule_date"].dt.tz is not None else df["schedule_date"]
    df["created_at"] = df["created_at"].dt.tz_localize(None) if df["created_at"].dt.tz is not None else df["created_at"]
    df["days_in_advance"] = (df["schedule_date"] - df["created_at"].dt.normalize()).dt.days.clip(lower=0)
    df["day_of_week"]     = df["schedule_date"].dt.dayofweek
    df["month"]           = df["schedule_date"].dt.month
    df["is_lunch"]        = (df["schedule_type"] == "lunch").astype(int)
    df["is_weekend"]      = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_monday"]       = (df["day_of_week"] == 0).astype(int)

    col = "meal_option" if "meal_option" in df.columns else "meal_type" if "meal_type" in df.columns else None
    df["meal_type_enc"] = df[col].map(MEAL_TYPE_ENC).fillna(2).astype(int) if col else 2

    # Prior de no-show por dia da semana (usado quando o utilizador tem < 3 registos)
    df["prior_noshow"] = df["day_of_week"].map(PRIOR_NOSHOW_BY_WEEKDAY).fillna(0.2)

    # Stats do utilizador — com fallback para o prior quando há poucos dados
    user_stats = df.groupby("user_cpf").agg(
        _noshow_sum=("is_noshow", "sum"),
        _total=("is_noshow", "count"),
    ).reset_index()
    user_stats["user_noshow_rate"]         = user_stats["_noshow_sum"] / user_stats["_total"].replace(0, 1)
    user_stats["user_noshow_count"]        = user_stats["_noshow_sum"]
    user_stats["user_total_schedules"]     = user_stats["_total"]
    user_stats["user_reliability_score"]   = 1 - user_stats["user_noshow_rate"]

    df = df.merge(user_stats[["user_cpf", "user_noshow_rate", "user_noshow_count",
                               "user_total_schedules", "user_reliability_score"]],
                  on="user_cpf", how="left")

    # Blending: menos dados do utilizador → mais peso no prior
    BLEND_THRESHOLD = 5
    df["blend_weight"] = (df["user_total_schedules"] / BLEND_THRESHOLD).clip(0, 1)
    df["user_noshow_rate"] = (
        df["blend_weight"] * df["user_noshow_rate"] +
        (1 - df["blend_weight"]) * df["prior_noshow"]
    )

    df = df.fillna({"user_noshow_rate": 0.2, "user_noshow_count": 0,
                    "user_total_schedules": 0, "user_reliability_score": 0.8})

    logger.info(f"Features de no-show geradas: {len(df)} registos.")
    return df
