from fastapi import APIRouter
from db import get_connection
from logger import logger

router = APIRouter()

@router.get("/health")
def health():
    db_ok = False
    db_error = None
    schedules_count = 0

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                db_ok = True
                # Conta agendamentos para confirmar acesso real aos dados
                cur.execute("SELECT COUNT(*) AS total FROM schedule")
                row = cur.fetchone()
                schedules_count = row["total"] if row else 0
    except Exception as e:
        db_error = str(e)
        logger.error(f"Health check DB falhou: {e}")

    return {
        "status": "ok" if db_ok else "degraded",
        "service": "SmartRU AI",
        "db": {
            "connected": db_ok,
            "schedules_count": schedules_count,
            "error": db_error,
        },
    }
