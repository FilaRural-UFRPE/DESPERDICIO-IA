import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "smartru-postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "smart_ru")
POSTGRES_USER = os.getenv("POSTGRES_USER", "smart_ru_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "y7wHtrQQWXPjXvHezmdQ



")
SMARTRU_API_URL = os.getenv("SMARTRU_API_URL", "https://semdesperdicio.smartru.com.br/api")
SMARTRU_ADMIN_API_KEY = os.getenv("SMARTRU_ADMIN_API_KEY", "")
MODEL_DIR = os.getenv("MODEL_DIR", "data/models")
MIN_DATA_DAYS = int(os.getenv("MIN_DATA_DAYS", 30))
RETRAIN_HOUR = int(os.getenv("RETRAIN_HOUR", 3))
