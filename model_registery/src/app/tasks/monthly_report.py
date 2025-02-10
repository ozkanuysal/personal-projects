from celery import shared_task
from ..services.report import ReportService
from ..services.notification import NotificationService
from sqlalchemy.orm import Session
from ..crud.db import SessionLocal
from ..services.model_registry import model_registry_service
from ..services.inference import inference_service
from ..crud.logger import AppLogger

from ..models.model import Deployment, Model, ModelVersion
from ..models.user import User

logger = AppLogger(__name__).get_logger()
@shared_task(name='monthly_report_task')
def monthly_report_task():
    logger.info("Monthly report task started.")
    db: Session = SessionLocal()

    model_name, version_id, model_path = (db.query(Model.name, ModelVersion.version, ModelVersion.model_path)
                                          .join(ModelVersion, Model.id == ModelVersion.model_id)
                                          .join(Deployment, ModelVersion.id == Deployment.model_version_id)
                                          .filter(Deployment.status == 'deployed')
                                          .all())[0]
    model_version_record, model_file_path = model_registry_service.get_model(model_name, version_id)
    artifact_file_path = model_registry_service.get_artifacts(model_name, version_id)[0]
    inference_service.report_inference(model_file_path, artifact_file_path)

    logger.info("Monthly report task finished.")