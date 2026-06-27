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


def _parse_schedules(raw: list) -> list:
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
    return records


def collect_schedules(days_back: int = 90) -> pd.DataFrame:
    """
    Coleta agendamentos dos últimos N dias via API do SmartRU.
    O endpoint requer parâmetro date ou user_cpf.
    """
    all_records = []
    today = date.today()

    for i in range(days_back):
        target = today - timedelta(days=i)
        date_str = f"{target.day:02d}/{target.month:02d}/{target.year}"
        data = _get("/schedule/all", params={"date": date_str})
        raw  = data.get("data", [])
        if raw:
            all_records.extend(_parse_schedules(raw))

    if not all_records:
        logger.warning("collect_schedules: sem agendamentos nos últimos %d dias.", days_back)
        return pd.DataFrame()

    df = pd.DataFrame(all_records).drop_duplicates(subset=["id"])
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


def check_api_connection() -> bool:
    """Verifica se a API do SmartRU está acessível."""
    try:
        today = date.today()
        date_str = f"{today.day:02d}/{today.month:02d}/{today.year}"
        res = requests.get(
            f"{SMARTRU_API_URL}/schedule/all",
            headers=_headers(),
            params={"date": date_str},
            timeout=5,
        )
        return res.status_code < 500
    except Exception:
        return False
