from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import health
import demand
import noshow
import patterns
import recommendations
from scheduler import start_scheduler
from trainer import train_all
from logger import logger

# Treina os modelos ANTES de registar os routers
logger.info("Treino inicial antes do arranque do servidor...")
try:
    train_all()
    logger.info("Treino inicial concluído.")
except Exception as e:
    logger.error(f"Erro no treino inicial: {e}")

app = FastAPI(
    title="SmartRU AI",
    description="Microserviço de IA para otimização do Restaurante Universitário da UFRPE",
    version="1.0.0",
)

# CORS — permite chamadas do frontend SmartRU
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://semdesperdicio.smartru.com.br",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(demand.router, prefix="/api/demand", tags=["demand"])
app.include_router(noshow.router, prefix="/api/noshow", tags=["noshow"])
app.include_router(patterns.router, prefix="/api/patterns", tags=["patterns"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])

@app.on_event("startup")
async def startup():
    logger.info("SmartRU AI iniciando...")
    start_scheduler()
    logger.info("Scheduler de treino automático ativo.")
