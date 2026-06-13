from fastapi import FastAPI
import health
import demand
import noshow
import patterns
import recommendations
from scheduler import start_scheduler
from logger import logger

app = FastAPI(
    title="SmartRU AI",
    description="Microserviço de IA para otimização do Restaurante Universitário da UFRPE",
    version="1.0.0",
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
