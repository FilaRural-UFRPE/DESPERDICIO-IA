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

"""
Adiciona este código ao final do main.py do DESPERDICIO-IA.
Endpoint temporário para debug — REMOVER em produção final.
"""

@app.get("/debug/schedules-count")
def debug_schedules_count():
    from collector import collect_schedules
    df = collect_schedules()
    return {"total_agendamentos": len(df)}


@app.get("/debug/users-count")
def debug_users_count():
    import requests, os
    headers = {"Authorization": f"Bearer {os.environ.get('ADMIN_API_KEY', '')}"}
    r = requests.get(
        "https://semdesperdicio.smartru.com.br/api/users",
        headers=headers,
    )
    data = r.json()
    return {"total_cadastrados": len(data.get("data", []))}


@app.get("/debug/menu-dishes")
def debug_menu_dishes():
    from collector import collect_menu_dishes
    df = collect_menu_dishes()
    return {
        "total_menus_com_pratos": len(df),
        "sample": df.head(3).to_dict(orient="records") if not df.empty else [],
    }
