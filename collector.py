import os
import requests
import pandas as pd
from datetime import date, timedelta
from logger import logger

SMARTRU_API_URL = os.environ.get("SMARTRU_API_URL", "https://semdesperdicio.smartru.com.br/api")
ADMIN_API_KEY   = os.environ.get("ADMIN_API_KEY", "")

def _headers():
    return {
        "Authorization": f"Bearer {ADMIN_API_KEY}",
        "Content-Type": "application/json",
    }

def _get(endpoint: str, params: dict = None) -> dict:
    try:
        res = requests.get(
            f"{SMARTRU_API_URL}{endpoint}",
            headers=_headers(),
            params=params,
            timeout=10,
        )
        if res.status_code == 404:
            return {"data": []}
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error(f"Erro ao chamar API SmartRU [{endpoint}]: {e}")
        return {}


def collect_schedules() -> pd.DataFrame:
    """Coleta todos os agendamentos via API do SmartRU."""
    data = _get("/schedule/all")
    raw  = data.get("data", [])

    if not raw:
        logger.warning("collect_schedules: sem agendamentos.")
        return pd.DataFrame()

    records = []
    for s in raw:
        records.append({
            "id":             s.get("id"),
            "user_cpf":       s.get("user_cpf"),
            "schedule_type":  s.get("schedule_type"),
            "schedule_date":  s.get("schedule_date"),
            "estimated_time": s.get("estimated_time"),
            "status":         s.get("status", "AGENDADO"),
            "meal_option":    s.get("meal_option") or s.get("meal_type", "essencial"),
            "created_at":     s.get("created_at"),
            "is_noshow":      1 if s.get("status") in ("CANCELADO", "AGENDADO") else 0,
        })

    df = pd.DataFrame(records).drop_duplicates(subset=["id"])
    logger.info(f"collect_schedules: {len(df)} registos recolhidos.")
    return df


def collect_daily_demand() -> pd.DataFrame:
    df = collect_schedules()
    if df.empty:
        return pd.DataFrame()

    df["schedule_date"] = pd.to_datetime(df["schedule_date"]).dt.date.astype(str)

    agg = df.groupby(["schedule_date", "schedule_type", "meal_option"]).agg(
        total_agendados=("id", "count"),
        confirmed=("is_noshow", lambda x: (x == 0).sum()),
        noshow_count=("is_noshow", "sum"),
    ).reset_index()

    agg.rename(columns={"schedule_type": "meal_type"}, inplace=True)
    logger.info(f"collect_daily_demand: {len(agg)} registos agregados.")
    return agg


def collect_menu_dishes() -> pd.DataFrame:
    """
    Recolhe os pratos extraídos do cardápio pelo Menu Analyzer.
    Depende do campo dishes JSONB na tabela menu.
    Retorna DataFrame vazio se o campo não existir ainda.
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", ""),
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            dbname=os.environ.get("POSTGRES_DB", ""),
            user=os.environ.get("POSTGRES_USER", ""),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            cursor_factory=RealDictCursor,
        )

        with conn:
            with conn.cursor() as cur:
                # Verifica se o campo dishes existe
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'menu' AND column_name = 'dishes'
                """)
                if not cur.fetchone():
                    logger.warning("collect_menu_dishes: campo dishes ainda não existe na tabela menu.")
                    return pd.DataFrame()

                cur.execute("""
                    SELECT 
                        id,
                        uploaded_at::date AS menu_date,
                        dishes
                    FROM menu
                    WHERE dishes IS NOT NULL
                    ORDER BY uploaded_at DESC
                """)
                rows = cur.fetchall()

        if not rows:
            logger.warning("collect_menu_dishes: sem pratos extraídos ainda.")
            return pd.DataFrame()

        records = []
        for row in rows:
            dishes = row["dishes"]
            if not dishes:
                continue

            # Extrai features dos pratos
            lunch = dishes.get("lunch", {})
            dinner = dishes.get("dinner", {})

            lunch_dishes  = lunch.get("dishes", {}) if isinstance(lunch, dict) else {}
            dinner_dishes = dinner.get("dishes", {}) if isinstance(dinner, dict) else {}

            lunch_tipos  = lunch.get("tipos_refeicao", {}) if isinstance(lunch, dict) else {}
            dinner_tipos = dinner.get("tipos_refeicao", {}) if isinstance(dinner, dict) else {}

            records.append({
                "menu_id":              row["id"],
                "menu_date":            str(row["menu_date"]),
                # Features do almoço
                "lunch_has_chicken":    _has_keyword(lunch_dishes, ["frango", "chicken", "galinha"]),
                "lunch_has_beef":       _has_keyword(lunch_dishes, ["carne", "beef", "boi", "picanha", "alcatra"]),
                "lunch_has_fish":       _has_keyword(lunch_dishes, ["peixe", "fish", "tilápia", "atum", "salmão"]),
                "lunch_has_vegetarian": _has_keyword(lunch_dishes, ["vegetariano", "vegano", "vegan", "kibe vegano"]),
                "lunch_num_options":    _count_options(lunch_dishes),
                "lunch_has_select":     bool(lunch_tipos.get("select")),
                "lunch_has_leve_sabor": bool(lunch_tipos.get("leve_sabor")),
                # Features do jantar
                "dinner_has_chicken":    _has_keyword(dinner_dishes, ["frango", "chicken", "galinha"]),
                "dinner_has_beef":       _has_keyword(dinner_dishes, ["carne", "beef", "boi"]),
                "dinner_has_fish":       _has_keyword(dinner_dishes, ["peixe", "fish", "tilápia"]),
                "dinner_has_vegetarian": _has_keyword(dinner_dishes, ["vegetariano", "vegano"]),
                "dinner_num_options":    _count_options(dinner_dishes),
            })

        df = pd.DataFrame(records)
        logger.info(f"collect_menu_dishes: {len(df)} cardápios com pratos extraídos.")
        return df

    except Exception as e:
        logger.warning(f"collect_menu_dishes: erro ao recolher pratos — {e}")
        return pd.DataFrame()


def _has_keyword(dishes: dict, keywords: list) -> int:
    """Verifica se algum prato contém uma das palavras-chave."""
    if not dishes:
        return 0
    all_items = []
    for items in dishes.values():
        if isinstance(items, list):
            all_items.extend([str(i).lower() for i in items])
    return int(any(kw in item for item in all_items for kw in keywords))


def _count_options(dishes: dict) -> int:
    """Conta o número total de pratos disponíveis."""
    if not dishes:
        return 0
    return sum(len(v) for v in dishes.values() if isinstance(v, list))


def check_api_connection() -> bool:
    """Verifica se a API do SmartRU está acessível."""
    try:
        res = requests.get(
            f"{SMARTRU_API_URL}/schedule/all",
            headers=_headers(),
            timeout=5,
        )
        return res.status_code < 500
    except Exception:
        return False
