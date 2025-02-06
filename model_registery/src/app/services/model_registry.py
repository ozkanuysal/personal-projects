import os
import uuid
from typing import List, Tuple

from fastapi import HTTPException
from fastapi import UploadFile
from sqlalchemy.orm import Session

from .minio import minio_service
from ..crud.db import SessionLocal
from ..models.model import Model, ModelVersion, Deployment


class ModelRegistryService:
    """
    A service class for managing models, model versions, and deployments.
    """

    def __init__(self, db: Session):
        """
        Initialize the ModelRegistryService with a database session.

        :param db: The database session.
        """
        self.db = db

    def create_model(self, name: str, description: str, author_id: int) -> Model:
        """
        Create a new model record in the database.

        :param name: The name of the model.
        :param description: The description of the model.
        :param author_id: The ID of the author of the model.
        :return: The created model record.
        """
        model = Model(name=name, description=description, author_id=author_id)
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    def create_model_version(self, model_id: int, version: str, model_path: str, metrics: dict) -> ModelVersion:
        """
        Create a new model version record in the database.

        :param model_id: The ID of the model.
        :param version: The version of the model.
        :param model_path: The path of the model file in storage.
        :param metrics: The metrics associated with the model version.
        :return: The created model version record.
        """
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_path=model_path,
            metrics=metrics
        )
        self.db.add(model_version)
        self.db.commit()
        self.db.refresh(model_version)
        return model_version

    def create_deployment(self, version_id: int, environment: str = "production") -> Deployment:
        """
        Create a new deployment record in the database.

        :param version_id: The ID of the model version.
        :param environment: The environment where the deployment is being made (default is "production").
        :return: The created deployment record.
        """
        deployment = Deployment(model_version_id=version_id, environment=environment, status="deployed")
        self.db.add(deployment)
        self.db.commit()
        self.db.refresh(deployment)
        return deployment

    def upload_model_and_artifacts(self, model_name: str, model_version: str, model_file: UploadFile,
                                   artifact_files: List[UploadFile]) -> Tuple[str, List[str]]:
        """
        Upload a model file and its associated artifacts to storage.

        :param model_name: The name of the model.
        :param model_version: The version of the model.
        :param model_file: The model file to be uploaded.
        :param artifact_files: The list of artifact files to be uploaded.
        :return: The path of the uploaded model file and the paths of the uploaded artifact files.
        """
        # Save the model file
        model_file_path = f"/tmp/{str(uuid.uuid4())}_{model_file.filename}"
        with open(model_file_path, "wb") as buffer:
            buffer.write(model_file.file.read())
        model_path = f"{model_name}/{model_version}/{model_file.filename}"
        minio_service.upload_file(model_file_path, model_path)
        os.remove(model_file_path)

        artifact_paths = []
        for artifact_file in artifact_files:
            artifact_file_path = f"/tmp/{str(uuid.uuid4())}_{artifact_file.filename}"
            with open(artifact_file_path, "wb") as buffer:
                buffer.write(artifact_file.file.read())

            artifact_path = f"{model_name}/{model_version}/artifacts/{artifact_file.filename}"
            minio_service.upload_file(artifact_file_path, artifact_path)
            artifact_paths.append(artifact_path)
            os.remove(artifact_file_path)

        return model_path, artifact_paths

    def get_model(self, model_name: str, model_version: str) -> Tuple[ModelVersion, str]:
        """
        Retrieve a model version record and download the associated model file.

        :param model_name: The name of the model.
        :param model_version: The version of the model.
        :return: The model version record and the path of the downloaded model file.
        """
        model_version_record = self.db.query(ModelVersion).join(Model).filter(
            Model.name == model_name,
            ModelVersion.version == model_version
        ).first()

        if not model_version_record:
            raise HTTPException(status_code=404, detail="Model version not found")

        # Create a temporary path to download the model file
        model_file_path = f"/tmp/{str(uuid.uuid4())}_{os.path.basename(model_version_record.model_path)}"
        minio_service.download_file(model_version_record.model_path, model_file_path)

        return model_version_record, model_file_path

    def get_artifacts(self, model_name: str, model_version: str) -> List[str]:
        """
        Retrieve the paths of the artifacts associated with a model version.

        :param model_name: The name of the model.
        :param model_version: The version of the model.
        :return: The paths of the downloaded artifact files.
        """
        model_version_record = self.db.query(ModelVersion).join(Model).filter(
            Model.name == model_name,
            ModelVersion.version == model_version
        ).first()

        if not model_version_record:
            raise HTTPException(status_code=404, detail="Model version not found")

        # Get the model path, which should be used to identify the directory in MinIO
        model_path = model_version_record.model_path
        artifact_dir = "/".join(model_path.split("/")[:-1] + ["artifacts"])  # Directory where artifacts are stored

        # List all files in the artifact directory from MinIO
        try:
            artifact_files = minio_service.list_files(artifact_dir)  # Replace with actual MinIO list function
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error accessing MinIO: {str(e)}")

        if not artifact_files:
            raise HTTPException(status_code=404, detail="No artifacts found for the specified model version")

        artifact_file_paths = []
        for artifact_file in artifact_files:
            artifact_file_path = f"/tmp/{str(uuid.uuid4())}_{os.path.basename(artifact_file)}"
            try:
                minio_service.download_file(artifact_file, artifact_file_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading file from MinIO: {str(e)}")

            # Append the file stream to the list
            artifact_file_paths.append(artifact_file_path)

        return artifact_file_paths


model_registry_service = ModelRegistryService(SessionLocal())