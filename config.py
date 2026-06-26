import os

POSTGRES_HOST     = os.environ["POSTGRES_HOST"]
POSTGRES_PORT     = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB       = os.environ["POSTGRES_DB"]
POSTGRES_USER     = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]

SMARTRU_API_URL       = os.environ.get("SMARTRU_API_URL", "https://semdesperdicio.smartru.com.br/api")
SMARTRU_ADMIN_API_KEY = os.environ.get("SMARTRU_ADMIN_API_KEY", "")

MODEL_DIR     = os.environ.get("MODEL_DIR", "data/models")
MIN_DATA_DAYS = int(os.environ.get("MIN_DATA_DAYS", 30))
RETRAIN_HOUR  = int(os.environ.get("RETRAIN_HOUR", 3))
