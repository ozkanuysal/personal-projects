import json
import os
import pickle
import uuid

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .minio import minio_service
from ..models.data import Customer, Transaction
from ..models.model import Model, ModelVersion, Deployment
from ..models.user import User
from ..utils.path import get_data_path, get_model_file_path


class DataIngestionService:
    """
    A class to handle data ingestion tasks for a machine learning application.

    Attributes:
    None

    Methods:
    ingest_models(db: Session) -> None:
        Ingests initial model data into the database.

    ingest_model_versions(db: Session) -> None:
        Ingests initial model version data into the database.

    ingest_deployments(db: Session) -> None:
        Ingests initial deployment data into the database.

    ingest_minio() -> None:
        Uploads initial model and artifact files to MinIO.

    ingest_users(db: Session) -> None:
        Ingests initial user data into the database.

    ingest_customers(db: Session) -> None:
        Ingests initial customer data into the database.

    ingest_transactions(db: Session) -> None:
        Ingests initial transaction data into the database.

    create_initial_data(db: Session) -> None:
        Calls all the data ingestion methods to create initial data.
    """

    @staticmethod
    def ingest_models(db: Session) -> None:
        """Ingests initial model data into the database."""
        if db.query(Model).count() == 0:
            model = Model(name="test_model", description="This is a test model.", author_id=1)
            db.add(model)
            db.commit()

    @staticmethod
    def ingest_model_versions(db: Session) -> None:
        """Ingests initial model version data into the database."""
        if db.query(ModelVersion).count() == 0:
            model = ModelVersion(model_id=1, version="1", model_path="test_model/1/test_model.pkl", metrics={})
            db.add(model)
            db.commit()

    @staticmethod
    def ingest_deployments(db: Session) -> None:
        """Ingests initial deployment data into the database."""
        if db.query(Deployment).count() == 0:
            deployment = Deployment(model_version_id=1, environment="production", status="deployed")
            db.add(deployment)
            db.commit()

    @staticmethod
    def ingest_minio() -> None:
        """Uploads initial model and artifact files to MinIO."""
        notebooks_path = get_model_file_path()
        model_file_path = os.path.join(notebooks_path, "models/Model_0.pkl")
        artifact_file_path = os.path.join(notebooks_path, "config.json")

        # Load the model using pickle
        model = pd.read_pickle(model_file_path)

        # Serialize the model
        serialized_model_path = f"/tmp/{str(uuid.uuid4())}_test_model.pkl"
        with open(serialized_model_path, "wb") as buffer:
            pickle.dump(model, buffer)

        model_path = "test_model/1/test_model.pkl"
        minio_service.upload_file(serialized_model_path, model_path)
        os.remove(serialized_model_path)

        with open(artifact_file_path, "r") as file:
            artifact_content = file.read()

        artifact_data = json.loads(artifact_content)

        serialized_artifact_path = f"/tmp/{str(uuid.uuid4())}_config.json"
        with open(serialized_artifact_path, "w") as buffer:
            json.dump(artifact_data, buffer)

        artifact_path = "test_model/1/artifacts/config.json"
        minio_service.upload_file(serialized_artifact_path, artifact_path)
        os.remove(serialized_artifact_path)

    @staticmethod
    def ingest_users(db: Session) -> None:
        """Ingests initial user data into the database."""
        if db.query(User).count() == 0:
            user1 = User(username="admin", email="admin@example.com", role="admin")
            user1.set_password("password123")  # Hash the password

            user2 = User(username="developer", email="dev@example.com", role="developer")
            user2.set_password("password123")
            db.add(user1)
            db.add(user2)

            db.commit()

    @staticmethod
    def ingest_customers(db: Session) -> None:
        """Ingests initial customer data into the database."""
        if db.query(Customer).count() == 0:

            file_path = get_data_path()
            df = pd.read_csv(file_path)
            df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

            for col in df.columns:
                df[col] = df[col].replace(np.nan, None).replace(np.NaN, None)

            customer_table_data = df.drop_duplicates(subset=["customer_id"], keep="last")[
                ["customer_id", "age", "gender", "annual_income"]]

            for index, row in customer_table_data.iterrows():
                customer = Customer(
                    customer_id=row['customer_id'],
                    age=row['age'],
                    gender=row['gender'],
                    annual_income=row['annual_income']
                )
                db.add(customer)
            db.commit()

    @staticmethod
    def ingest_transactions(db: Session) -> None:
        """Ingests initial transaction data into the database."""
        if db.query(Transaction).count() == 0:
            file_path = get_data_path()
            df = pd.read_csv(file_path)

            transaction_table_data = df[["customer_id", "purchase_amount", "purchase_date"]]

            for index, row in transaction_table_data.iterrows():
                transaction = Transaction(
                    customer_id=row['customer_id'],
                    purchase_amount=row['purchase_amount'],
                    purchase_date=row['purchase_date']
                )
                db.add(transaction)
            db.commit()

    @staticmethod
    def create_initial_data(db: Session) -> None:
        """Calls all the data ingestion methods to create initial data."""
        DataIngestionService.ingest_users(db)
        DataIngestionService.ingest_models(db)
        DataIngestionService.ingest_customers(db)
        DataIngestionService.ingest_transactions(db)
        DataIngestionService.ingest_model_versions(db)
        DataIngestionService.ingest_deployments(db)
        DataIngestionService.ingest_models(db)
        DataIngestionService.ingest_minio()


data_ingestion_service = DataIngestionService