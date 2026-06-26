import pandas as pd
import numpy as np
from fastapi import APIRouter
from collector import collect_schedules
from processor import MEAL_TYPE_ENC, PRIOR_NOSHOW_BY_WEEKDAY, PRIOR_DEMAND_BY_WEEKDAY

router = APIRouter()

DAY_NAMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

MEAL_OPTION_LABELS = {
    "select":      "👑 Select",
    "leve_sabor":  "🥗 Leve Sabor",
    "essencial":   "🍱 Essencial",
    "vegetariano": "🌿 Vegetariano",
}


def _get_meal_col(df: pd.DataFrame) -> str:
    if "meal_option" in df.columns:
        return "meal_option"
    if "meal_type" in df.columns:
        return "meal_type"
    return None


@router.get("/weekly")
def weekly_patterns():
    """
    Padrões semanais de demanda e no-show.
    Com poucos dados: retorna prior + dados disponíveis mesclados.
    """
    df = collect_schedules()

    # Prior sempre disponível
    prior_by_day = [
        {
            "day": DAY_NAMES[i],
            "day_index": i,
            "total": PRIOR_DEMAND_BY_WEEKDAY[i],
            "noshow_rate": PRIOR_NOSHOW_BY_WEEKDAY[i],
            "source": "prior",
        }
        for i in range(7)
    ]

    if df.empty or len(df) < 5:
        busiest_prior = max(PRIOR_DEMAND_BY_WEEKDAY, key=PRIOR_DEMAND_BY_WEEKDAY.get)
        return {
            "data_available": False,
            "note": f"Apenas {len(df)} agendamentos. Mostrando estimativas típicas de RU.",
            "busiest_day": DAY_NAMES[busiest_prior],
            "lunch_total": 0,
            "dinner_total": 0,
            "overall_noshow_rate": 0.2,
            "by_day": prior_by_day,
        }

    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["day_of_week"]   = df["schedule_date"].dt.dayofweek

    by_day = df.groupby("day_of_week").agg(
        total=("id", "count"),
        noshow_rate=("is_noshow", "mean"),
    ).reset_index()

    # Mescla dados reais com prior (quanto mais dados, menos prior)
    merged = []
    for i in range(7):
        row = by_day[by_day["day_of_week"] == i]
        if row.empty:
            merged.append({
                "day": DAY_NAMES[i],
                "day_index": i,
                "total": PRIOR_DEMAND_BY_WEEKDAY[i],
                "noshow_rate": PRIOR_NOSHOW_BY_WEEKDAY[i],
                "source": "prior",
            })
        else:
            obs_total  = int(row.iloc[0]["total"])
            obs_noshow = float(row.iloc[0]["noshow_rate"])
            weight     = min(obs_total / 20, 1.0)
            merged.append({
                "day": DAY_NAMES[i],
                "day_index": i,
                "total": obs_total,
                "noshow_rate": round(weight * obs_noshow + (1 - weight) * PRIOR_NOSHOW_BY_WEEKDAY[i], 3),
                "source": "observed" if weight > 0.5 else "blended",
            })

    busiest_idx = max(merged, key=lambda x: x["total"])["day_index"]
    col         = _get_meal_col(df)
    most_popular = df[col].mode()[0] if col and not df.empty else "essencial"

    return {
        "data_available": True,
        "total_records": len(df),
        "busiest_day": DAY_NAMES[busiest_idx],
        "lunch_total": int(len(df[df["schedule_type"] == "lunch"])),
        "dinner_total": int(len(df[df["schedule_type"] == "dinner"])),
        "overall_noshow_rate": round(float(df["is_noshow"].mean()), 3),
        "most_popular_meal": MEAL_OPTION_LABELS.get(most_popular, most_popular),
        "peak_hour": _get_peak_hour(df),
        "noshow_by_day": {str(d["day_index"]): d["noshow_rate"] for d in merged},
        "by_day": merged,
    }


def _get_peak_hour(df: pd.DataFrame) -> str:
    if "estimated_time" not in df.columns or df["estimated_time"].isna().all():
        return "12:00"
    try:
        hours = pd.to_datetime(df["estimated_time"], format="%H:%M:%S", errors="coerce").dt.hour
        peak  = int(hours.mode()[0]) if not hours.isna().all() else 12
        return f"{peak:02d}:00"
    except Exception:
        return "12:00"


@router.get("/meal-options")
def meal_option_patterns():
    """
    Distribuição histórica por tipo de refeição.
    Retorna distribuição padrão quando há poucos dados.
    """
    DEFAULT = {
        "select":      {"percentage": 20, "label": MEAL_OPTION_LABELS["select"],      "source": "prior"},
        "leve_sabor":  {"percentage": 30, "label": MEAL_OPTION_LABELS["leve_sabor"],  "source": "prior"},
        "essencial":   {"percentage": 45, "label": MEAL_OPTION_LABELS["essencial"],   "source": "prior"},
        "vegetariano": {"percentage": 5,  "label": MEAL_OPTION_LABELS["vegetariano"], "source": "prior"},
    }

    df  = collect_schedules()
    col = _get_meal_col(df)

    if df.empty or col is None or len(df) < 10:
        return {
            "data_available": False,
            "total_records": len(df),
            "note": "Poucos dados. Distribuição típica estimada.",
            "distribution": DEFAULT,
        }

    total  = len(df)
    counts = df[col].value_counts()
    dist   = {}

    for option, label_str in MEAL_OPTION_LABELS.items():
        obs_count = int(counts.get(option, 0))
        obs_pct   = round((obs_count / total) * 100, 1)
        prior_pct = DEFAULT[option]["percentage"]
        weight    = min(total / 50, 1.0)
        final_pct = round(weight * obs_pct + (1 - weight) * prior_pct, 1)
        dist[option] = {
            "count":      obs_count,
            "percentage": final_pct,
            "label":      label_str,
            "source":     "observed" if weight > 0.5 else "blended",
        }

    most_popular = max(dist, key=lambda x: dist[x]["percentage"])
    return {
        "data_available": True,
        "total_records": total,
        "most_popular": MEAL_OPTION_LABELS[most_popular],
        "distribution": dist,
    }


@router.get("/user/{cpf}")
def user_patterns(cpf: str):
    df = collect_schedules()

    if df.empty:
        return {
            "user_cpf": cpf,
            "total_schedules": 0,
            "note": "Sem dados disponíveis.",
        }

    user_df = df[df["user_cpf"] == cpf]

    if user_df.empty:
        return {
            "user_cpf": cpf,
            "total_schedules": 0,
            "note": "Utilizador sem histórico de agendamentos.",
        }

    col = _get_meal_col(user_df)
    preferred_option  = user_df[col].mode()[0] if col and not user_df.empty else "essencial"
    preferred_meal    = user_df["schedule_type"].mode()[0] if not user_df.empty else "lunch"
    noshow_rate       = float(user_df["is_noshow"].mean()) if "is_noshow" in user_df.columns else 0.0
    reliability       = "alta" if noshow_rate < 0.1 else "média" if noshow_rate < 0.3 else "baixa"

    return {
        "user_cpf": cpf,
        "total_schedules": len(user_df),
        "confirmed": int((user_df["is_noshow"] == 0).sum()) if "is_noshow" in user_df.columns else len(user_df),
        "noshow_rate": round(noshow_rate, 3),
        "reliability": reliability,
        "preferred_meal": "🍽️ Almoço" if preferred_meal == "lunch" else "🌙 Jantar",
        "preferred_option": MEAL_OPTION_LABELS.get(preferred_option, preferred_option),
    }
