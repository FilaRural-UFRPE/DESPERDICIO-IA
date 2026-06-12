from apscheduler.schedulers.background import BackgroundScheduler
from trainer import train_all
from config import RETRAIN_HOUR
from logger import logger

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        train_all,
        trigger="cron",
        hour=RETRAIN_HOUR,
        minute=0,
        id="daily_retrain",
    )
    scheduler.start()
    logger.info(f"Treino automático agendado para {RETRAIN_HOUR}h diariamente.")
