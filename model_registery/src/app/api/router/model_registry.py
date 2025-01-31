import os
from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ...crud.db import get_db
from ...crud.logger import AppLogger
from ...models.user import User
from ...services.auth import auth_service
from ...services.model_registry import model_registry_service

logger = AppLogger(__name__).get_logger()

model_router = APIRouter(tags=["Model registry"])


@model_router.post("/upload_model")
async def upload_model(
        model_name: str,
        model_description: str,
        model_version: str,
        model_file: UploadFile = File(...),
        artifact_files: List[UploadFile] = File(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(auth_service.get_current_user)
):
    """
    This function handles the model upload process. It receives the model details, files, and user information,
    uploads the model and artifacts to the storage, creates model and version records in the database, and
    deploys the model.

    Parameters:
    - model_name (str): The name of the model.
    - model_description (str): The description of the model.
    - model_version (str): The version of the model.
    - model_file (UploadFile): The file containing the model.
    - artifact_files (List[UploadFile]): The files containing additional artifacts related to the model.
    - db (Session): The database session for interacting with the database.
    - current_user (User): The user making the request.

    Returns:
    - dict: A dictionary containing the model ID, model version ID, model path, artifact paths, and deployment status.

    Raises:
    - HTTPException: If an error occurs during the model upload process.
    """
    try:
        logger.info("User %s is uploading model: %s, version: %s", current_user, model_name, model_version)
        user = db.query(User).filter(User.username == current_user).first()
        model_path, artifact_paths = model_registry_service.upload_model_and_artifacts(model_name, model_version,
                                                                                        model_file, artifact_files)

        model = model_registry_service.create_model(model_name, model_description, user.id)
        model_version = model_registry_service.create_model_version(model.id, model_version, model_path,
                                                                     metrics={})
        deployment = model_registry_service.create_deployment(model_version.id)
        logger.info("Model upload successful for user: %s, model: %s, version: %s", current_user, model_name,
                    model_version)

        return {"model_id": model.id, "model_version_id": model_version.id, "model_path": model_path,
                "artifact_paths": artifact_paths, "status": deployment.status}
    except Exception as e:
        logger.error("Error uploading model: %s, version: %s for user: %s. Error: %s", model_name, model_version,
                     current_user, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@model_router.get("/get_model", tags=["Model registry"])
async def get_model(
        model_name: str,
        model_version: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(auth_service.get_current_user)
):
    """
    This function retrieves a specific model version from the model registry.

    Parameters:
    - model_name (str): The name of the model.
    - model_version (str): The version of the model.
    - db (Session): The database session for interacting with the database.
    - current_user (User): The user making the request.

    Returns:
    - FileResponse: A FileResponse object containing the requested model file.

    Raises:
    - HTTPException: If an error occurs during the retrieval process.
    """
    try:
        logger.info("User %s requested model: %s, version: %s", current_user, model_name, model_version)
        model_version_record, model_file_path = model_registry_service.get_model(model_name, model_version)
        logger.info("Successfully retrieved model: %s, version: %s for user: %s", model_name, model_version,
                    current_user)
        return FileResponse(model_file_path, filename=os.path.basename(model_file_path),
                            media_type='application/octet-stream')
    except Exception as e:
        logger.error("Error retrieving model: %s, version: %s for user: %s. Error: %s", model_name, model_version,
                     current_user, str(e))

        raise HTTPException(status_code=500, detail=str(e))