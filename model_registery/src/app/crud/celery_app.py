from celery import Celery
from celery.schedules import crontab
from .settings import settings
from datetime import timedelta
from ..tasks import *

celery_app = Celery(
    'app',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    result_expires=3600,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
)

celery_app.conf.beat_schedule = {
    'monthly_report_task': {
        'task': 'monthly_report_task',
        'schedule':  timedelta(seconds=180) if settings.CELERY_BEAT_SCHEDULE == "monthly" else timedelta(minutes=1),
    },
}