import pandas as pd
from fastapi import APIRouter
from collector import collect_schedules
from processor import PRIOR_NOSHOW_BY_WEEKDAY

router = APIRouter()

MEAL_OPTION_LABELS = {
    "select":      "👑 Select",
    "leve_sabor":  "🥗 Leve Sabor",
    "essencial":   "🍱 Essencial",
    "vegetariano": "🌿 Vegetariano",
}

MEAL_OPTIONS = list(MEAL_OPTION_LABELS.keys())

DAY_NAMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

DEFAULT_DISTRIBUTION = {
    "select":      {"percentage": 20, "label": "👑 Select",      "source": "prior"},
    "leve_sabor":  {"percentage": 30, "label": "🥗 Leve Sabor",  "source": "prior"},
    "essencial":   {"percentage": 45, "label": "🍱 Essencial",   "source": "prior"},
    "vegetariano": {"percentage": 5,  "label": "🌿 Vegetariano", "source": "prior"},
}


def _get_meal_col(df: pd.DataFrame) -> str:
    if "meal_option" in df.columns:
        return "meal_option"
    if "meal_type" in df.columns:
        return "meal_type"
    return None


@router.get("/menu")
def recommend_menu(date: str = None):
    """
    Recomenda quais tipos de refeição preparar mais.
    Funciona desde o primeiro agendamento — usa prior quando há poucos dados.
    """
    df  = collect_schedules()
    col = _get_meal_col(df)

    if df.empty or col is None:
        return {
            "method": "prior",
            "message": "Sem dados históricos ainda. Usando distribuição típica de RU.",
            "recommendations": DEFAULT_DISTRIBUTION,
        }

    # Filtrar por dia da semana se data fornecida
    if date:
        try:
            target = pd.to_datetime(date)
            df["schedule_date"] = pd.to_datetime(df["schedule_date"])
            df_day = df[df["schedule_date"].dt.dayofweek == target.dayofweek]
            if len(df_day) >= 5:
                df = df_day
        except Exception:
            pass

    total = len(df)
    counts = df[col].value_counts()
    weight = min(total / 30, 1.0)  # 0 = só prior, 1 = só observado

    recommendations = {}
    for option in MEAL_OPTIONS:
        obs_count = int(counts.get(option, 0))
        obs_pct   = round((obs_count / max(total, 1)) * 100, 1)
        prior_pct = DEFAULT_DISTRIBUTION[option]["percentage"]
        final_pct = round(weight * obs_pct + (1 - weight) * prior_pct, 1)

        recommendations[option] = {
            "count":      obs_count,
            "percentage": final_pct,
            "label":      MEAL_OPTION_LABELS[option],
            "source":     "observed" if weight > 0.7 else "blended" if weight > 0.2 else "prior",
        }

    most_popular = max(recommendations, key=lambda x: recommendations[x]["percentage"])
    method = "historical_data" if weight > 0.7 else "blended_prior" if weight > 0.2 else "prior"

    return {
        "method": method,
        "total_schedules_analyzed": total,
        "most_popular": MEAL_OPTION_LABELS[most_popular],
        "recommendations": recommendations,
        "suggestion": f"Prepare mais {MEAL_OPTION_LABELS[most_popular]} — é o tipo mais escolhido.",
    }


@router.get("/user/{cpf}")
def recommend_for_user(cpf: str):
    df  = collect_schedules()
    col = _get_meal_col(df)

    if df.empty or col is None:
        return {
            "user_cpf": cpf,
            "message": "Sem dados disponíveis. Recomendação padrão aplicada.",
            "preferred_option": "🍱 Essencial",
            "preferences": DEFAULT_DISTRIBUTION,
        }

    user_df = df[df["user_cpf"] == cpf]

    if user_df.empty or len(user_df) < 2:
        return {
            "user_cpf": cpf,
            "message": "Utilizador sem histórico suficiente. Recomendação baseada em padrões gerais.",
            "preferred_option": "🍱 Essencial",
            "preferences": DEFAULT_DISTRIBUTION,
        }

    total  = len(user_df)
    counts = user_df[col].value_counts()
    weight = min(total / 10, 1.0)

    preferences = {}
    for option in MEAL_OPTIONS:
        obs_count = int(counts.get(option, 0))
        obs_pct   = round((obs_count / max(total, 1)) * 100, 1)
        prior_pct = DEFAULT_DISTRIBUTION[option]["percentage"]
        final_pct = round(weight * obs_pct + (1 - weight) * prior_pct, 1)
        preferences[option] = {
            "count":      obs_count,
            "percentage": final_pct,
            "label":      MEAL_OPTION_LABELS[option],
        }

    preferred      = max(preferences, key=lambda x: preferences[x]["percentage"])
    preferred_meal = user_df["schedule_type"].mode()[0] if not user_df.empty else "lunch"

    return {
        "user_cpf": cpf,
        "total_schedules": total,
        "preferred_meal":   "🍽️ Almoço" if preferred_meal == "lunch" else "🌙 Jantar",
        "preferred_option": MEAL_OPTION_LABELS.get(preferred, preferred),
        "preferences":      preferences,
    }


@router.get("/weekly-trends")
def weekly_trends():
    """Tendências semanais por tipo de refeição."""
    df  = collect_schedules()
    col = _get_meal_col(df)

    if df.empty or col is None or len(df) < 5:
        return {
            "method": "prior",
            "note": f"Apenas {len(df) if not df.empty else 0} agendamentos. Mostrando estimativas típicas.",
            "trends": [
                {
                    "day": DAY_NAMES[i],
                    "total_schedules": 0,
                    "most_popular": "🍱 Essencial",
                    "distribution": {o: DEFAULT_DISTRIBUTION[o]["percentage"] for o in MEAL_OPTIONS},
                    "source": "prior",
                }
                for i in range(5)  # só dias úteis no prior
            ],
        }

    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["day_of_week"]   = df["schedule_date"].dt.dayofweek

    trends = []
    for day_idx in range(7):
        day_df = df[df["day_of_week"] == day_idx]
        if day_df.empty:
            continue

        total  = len(day_df)
        counts = day_df[col].value_counts()
        weight = min(total / 20, 1.0)

        distribution = {}
        for option in MEAL_OPTIONS:
            obs_pct   = round((int(counts.get(option, 0)) / max(total, 1)) * 100, 1)
            prior_pct = DEFAULT_DISTRIBUTION[option]["percentage"]
            distribution[option] = round(weight * obs_pct + (1 - weight) * prior_pct, 1)

        most_popular = max(distribution, key=distribution.get)
        trends.append({
            "day":            DAY_NAMES[day_idx],
            "total_schedules": total,
            "most_popular":   MEAL_OPTION_LABELS[most_popular],
            "distribution":   distribution,
            "source":         "observed" if weight > 0.5 else "blended",
        })

    return {
        "method": "historical_data",
        "total_records": len(df),
        "trends": trends,
    }
