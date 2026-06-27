from fastapi import APIRouter
from collector import check_api_connection, collect_schedules
from logger import logger

router = APIRouter()

@router.get("/health")
def health():
    api_ok = False
    schedules_count = 0
    error = None

    try:
        api_ok = check_api_connection()
        if api_ok:
            df = collect_schedules()
            schedules_count = len(df)
    except Exception as e:
        error = str(e)
        logger.error(f"Health check falhou: {e}")

    return {
        "status": "ok" if api_ok else "degraded",
        "service": "SmartRU AI",
        "api": {
            "connected": api_ok,
            "schedules_count": schedules_count,
            "error": error,
        },
    }
