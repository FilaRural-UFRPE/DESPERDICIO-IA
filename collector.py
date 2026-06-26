import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from logger import logger

# ─── Conexão com PostgreSQL ───────────────────────────
def _get_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST"),
        port=os.environ.get("POSTGRES_PORT", 5432),
        dbname=os.environ.get("POSTGRES_DB"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def _query(sql: str, params=None) -> pd.DataFrame:
    """Executa uma query e retorna um DataFrame. Retorna vazio em caso de erro."""
    try:
        with _get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        logger.error(f"Erro ao conectar ao banco: {e}")
        return pd.DataFrame()


# ─── Coleta de agendamentos individuais ───────────────
def collect_schedules() -> pd.DataFrame:
    """
    Retorna todos os agendamentos com status de no-show.
    No-show = agendado mas não confirmado.
    """
    sql = """
        SELECT
            s.id,
            s.user_cpf,
            s.schedule_type,
            s.schedule_date,
            s.estimated_time,
            s.status,
            s.meal_option,
            s.created_at,
            CASE WHEN s.status = 'CANCELADO' THEN 1
                 WHEN s.status = 'AGENDADO'  THEN 1
                 ELSE 0
            END AS is_noshow
        FROM schedule s
        ORDER BY s.schedule_date ASC
    """
    df = _query(sql)
    if not df.empty:
        logger.info(f"collect_schedules: {len(df)} registos.")
    return df


# ─── Coleta de demanda agregada por dia ───────────────
def collect_daily_demand() -> pd.DataFrame:
    """
    Retorna a demanda diária agregada por tipo de refeição.
    Usado para treinar o modelo de previsão de demanda.
    """
    sql = """
        SELECT
            schedule_date,
            schedule_type  AS meal_type,
            meal_option,
            COUNT(*)       AS total_agendados,
            SUM(CASE WHEN status = 'CONFIRMADO' THEN 1 ELSE 0 END) AS confirmed,
            SUM(CASE WHEN status IN ('CANCELADO', 'AGENDADO') THEN 1 ELSE 0 END) AS noshow_count
        FROM schedule
        GROUP BY schedule_date, schedule_type, meal_option
        ORDER BY schedule_date ASC
    """
    df = _query(sql)
    if not df.empty:
        logger.info(f"collect_daily_demand: {len(df)} registos agregados.")
    return df


# ─── Coleta de utilizadores ───────────────────────────
def collect_users() -> pd.DataFrame:
    """Retorna dados dos utilizadores para análise de perfil."""
    sql = """
        SELECT
            cpf,
            role AS type,
            created_at
        FROM "user"
        ORDER BY created_at ASC
    """
    return _query(sql)


# ─── Health check da conexão ──────────────────────────
def check_db_connection() -> bool:
    """Verifica se a conexão com o banco está funcionando."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        logger.info("Conexão com o banco OK.")
        return True
    except Exception as e:
        logger.error(f"Falha na conexão com o banco: {e}")
        return False
