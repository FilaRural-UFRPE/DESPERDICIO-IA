import pandas as pd
from db import get_connection
from logger import logger

def collect_schedules() -> pd.DataFrame:
    query = """
        SELECT
            s.id,
            s.user_cpf,
            s.schedule_type,
            s.meal_type,
            s.schedule_date,
            s.estimated_time,
            s.status,
            s.created_at,
            CASE WHEN s.status = 'CONFIRMADO' THEN 0 ELSE 1 END AS is_noshow
        FROM schedules s
        ORDER BY s.schedule_date ASC
    """
    try:
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Coletados {len(df)} agendamentos.")
        return df
    except Exception as e:
        logger.error(f"Erro ao coletar agendamentos: {e}")
        return pd.DataFrame()

def collect_daily_demand() -> pd.DataFrame:
    query = """
        SELECT
            schedule_date,
            schedule_type,
            meal_type,
            COUNT(*) AS total_agendados,
            COUNT(*) FILTER (WHERE status = 'CONFIRMADO') AS total_presentes,
            COUNT(*) FILTER (WHERE status != 'CONFIRMADO') AS total_noshow
        FROM schedules
        GROUP BY schedule_date, schedule_type, meal_type
        ORDER BY schedule_date ASC
    """
    try:
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Coletados {len(df)} registros de demanda diária.")
        return df
    except Exception as e:
        logger.error(f"Erro ao coletar demanda diária: {e}")
        return pd.DataFrame()
