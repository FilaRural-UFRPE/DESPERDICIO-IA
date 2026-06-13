import pandas as pd
from fastapi import APIRouter
from collector import collect_schedules

router = APIRouter()

MEAL_OPTION_LABELS = {
    "select": "👑 Select",
    "leve_sabor": "🥗 Leve Sabor",
    "essencial": "🍱 Essencial",
}

DAY_NAMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

def _get_meal_option_col(df: pd.DataFrame) -> str:
    """Retorna o nome correto da coluna de tipo de refeição."""
    if "meal_option" in df.columns:
        return "meal_option"
    if "meal_type" in df.columns:
        return "meal_type"
    return None

@router.get("/menu")
def recommend_menu(date: str = None):
    """
    Recomenda quais tipos de refeição preparar mais com base nas preferências históricas.
    Se não houver dados suficientes, retorna recomendação padrão.
    """
    df = collect_schedules()

    if df.empty:
        return {
            "method": "default",
            "message": "Sem dados históricos. Usando distribuição padrão.",
            "recommendations": {
                "select": {"percentage": 20, "label": "👑 Select"},
                "leve_sabor": {"percentage": 30, "label": "🥗 Leve Sabor"},
                "essencial": {"percentage": 50, "label": "🍱 Essencial"},
            }
        }

    col = _get_meal_option_col(df)
    if not col:
        return {"method": "default", "message": "Campo meal_option não encontrado."}

    # Filtrar por dia da semana se data fornecida
    if date:
        try:
            target = pd.to_datetime(date)
            df["schedule_date"] = pd.to_datetime(df["schedule_date"])
            df_day = df[df["schedule_date"].dt.dayofweek == target.dayofweek]
            if len(df_day) >= 10:
                df = df_day
        except Exception:
            pass

    total = len(df)
    if total < 10:
        return {
            "method": "default",
            "message": f"Apenas {total} agendamentos. Mínimo 10 para recomendação. Usando distribuição padrão.",
            "recommendations": {
                "select": {"percentage": 20, "label": "👑 Select"},
                "leve_sabor": {"percentage": 30, "label": "🥗 Leve Sabor"},
                "essencial": {"percentage": 50, "label": "🍱 Essencial"},
            }
        }

    counts = df[col].value_counts()
    recommendations = {}
    for option in ["select", "leve_sabor", "essencial"]:
        count = int(counts.get(option, 0))
        pct = round((count / total) * 100, 1)
        recommendations[option] = {
            "count": count,
            "percentage": pct,
            "label": MEAL_OPTION_LABELS.get(option, option),
        }

    # Ordenar por popularidade
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1]["percentage"], reverse=True)
    most_popular = sorted_recs[0][0] if sorted_recs else "essencial"

    return {
        "method": "historical_data",
        "total_schedules_analyzed": total,
        "most_popular": MEAL_OPTION_LABELS.get(most_popular, most_popular),
        "recommendations": recommendations,
        "suggestion": f"Prepare mais {MEAL_OPTION_LABELS.get(most_popular, most_popular)} — é o tipo mais escolhido pelos estudantes.",
    }

@router.get("/user/{cpf}")
def recommend_for_user(cpf: str):
    """
    Retorna as preferências de um estudante específico com base no seu histórico.
    """
    df = collect_schedules()

    if df.empty:
        return {"message": "Sem dados disponíveis"}

    user_df = df[df["user_cpf"] == cpf]
    if user_df.empty:
        return {"message": "Estudante sem histórico de agendamentos"}

    col = _get_meal_option_col(user_df)
    if not col:
        return {"message": "Campo meal_option não encontrado"}

    total = len(user_df)
    counts = user_df[col].value_counts()

    preferences = {}
    for option in ["select", "leve_sabor", "essencial"]:
        count = int(counts.get(option, 0))
        pct = round((count / total) * 100, 1) if total > 0 else 0
        preferences[option] = {
            "count": count,
            "percentage": pct,
            "label": MEAL_OPTION_LABELS.get(option, option),
        }

    preferred = counts.idxmax() if not counts.empty else "essencial"
    preferred_meal = user_df["schedule_type"].mode()[0] if not user_df.empty else "lunch"

    return {
        "user_cpf": cpf,
        "total_schedules": total,
        "preferred_meal": "🍽️ Almoço" if preferred_meal == "lunch" else "🌙 Jantar",
        "preferred_option": MEAL_OPTION_LABELS.get(preferred, preferred),
        "preferences": preferences,
    }

@router.get("/weekly-trends")
def weekly_trends():
    """
    Mostra quais tipos de refeição são mais populares por dia da semana.
    Útil para o gestor do RU planejar a produção semanal.
    """
    df = collect_schedules()

    if df.empty:
        return {"message": "Sem dados suficientes"}

    col = _get_meal_option_col(df)
    if not col:
        return {"message": "Campo meal_option não encontrado"}

    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["day_of_week"] = df["schedule_date"].dt.dayofweek

    trends = []
    for day_idx in range(7):
        day_df = df[df["day_of_week"] == day_idx]
        if day_df.empty:
            continue

        counts = day_df[col].value_counts()
        total = len(day_df)
        most_popular = counts.idxmax() if not counts.empty else "essencial"

        trends.append({
            "day": DAY_NAMES[day_idx],
            "total_schedules": total,
            "most_popular": MEAL_OPTION_LABELS.get(most_popular, most_popular),
            "distribution": {
                option: round((int(counts.get(option, 0)) / total) * 100, 1)
                for option in ["select", "leve_sabor", "essencial"]
            }
        })

    return {
        "method": "historical_data",
        "trends": trends,
    }
