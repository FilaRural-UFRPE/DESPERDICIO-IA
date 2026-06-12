import pandas as pd
from fastapi import APIRouter
from collector import collect_schedules

router = APIRouter()

DAY_NAMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

@router.get("/weekly")
def weekly_patterns():
    df = collect_schedules()
    if df.empty:
        return {"message": "Sem dados suficientes"}
    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["day_of_week"] = df["schedule_date"].dt.dayofweek
    by_day = df.groupby("day_of_week").agg(total=("id", "count"), noshow_rate=("is_noshow", "mean")).reset_index()
    busiest_day_idx = int(by_day.loc[by_day["total"].idxmax(), "day_of_week"])
    return {
        "busiest_day": DAY_NAMES[busiest_day_idx],
        "lunch_total": int(len(df[df["schedule_type"] == "lunch"])),
        "dinner_total": int(len(df[df["schedule_type"] == "dinner"])),
        "overall_noshow_rate": round(float(df["is_noshow"].mean()), 3),
        "by_day": [{"day": DAY_NAMES[int(row["day_of_week"])], "total": int(row["total"]), "noshow_rate": round(float(row["noshow_rate"]), 3)} for _, row in by_day.iterrows()],
    }

@router.get("/user/{cpf}")
def user_patterns(cpf: str):
    df = collect_schedules()
    if df.empty:
        return {"message": "Sem dados"}
    user_df = df[df["user_cpf"] == cpf]
    if user_df.empty:
        return {"message": "Usuário não encontrado"}
    return {
        "user_cpf": cpf,
        "total_schedules": int(len(user_df)),
        "confirmed": int((user_df["is_noshow"] == 0).sum()),
        "noshow_rate": round(float(user_df["is_noshow"].mean()), 3),
        "preferred_meal": user_df["schedule_type"].mode()[0],
    }
