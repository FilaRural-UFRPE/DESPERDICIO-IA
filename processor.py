import pandas as pd
from logger import logger

def process_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["day_of_week"] = df["schedule_date"].dt.dayofweek
    df["week_of_month"] = df["schedule_date"].dt.day // 7
    df["month"] = df["schedule_date"].dt.month
    df["is_lunch"] = (df["schedule_type"] == "lunch").astype(int)
    df["meal_type_enc"] = df["meal_type"].map(
        {"select": 0, "leve_sabor": 1, "essencial": 2}
    ).fillna(2)
    df = df.sort_values("schedule_date")
    df["rolling_7d"] = df["total_agendados"].rolling(7, min_periods=1).mean()
    df["rolling_14d"] = df["total_agendados"].rolling(14, min_periods=1).mean()
    logger.info("Features de demanda geradas.")
    return df

def process_noshow_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["days_in_advance"] = (df["schedule_date"] - df["created_at"].dt.normalize()).dt.days
    df["day_of_week"] = df["schedule_date"].dt.dayofweek
    df["is_lunch"] = (df["schedule_type"] == "lunch").astype(int)
    user_noshow = df.groupby("user_cpf")["is_noshow"].mean().rename("user_noshow_rate")
    df = df.merge(user_noshow, on="user_cpf", how="left")
    user_noshow_count = df.groupby("user_cpf")["is_noshow"].sum().rename("user_noshow_count")
    df = df.merge(user_noshow_count, on="user_cpf", how="left")
    logger.info("Features de no-show geradas.")
    return df
