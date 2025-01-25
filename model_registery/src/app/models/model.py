from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from ..crud.app import Base


class Model(Base):
    """
    Represents a machine learning model in the application.

    Attributes:
    - id (int): Unique identifier for the model.
    - name (str): Name of the model.
    - description (str): Description of the model.
    - author_id (int): Identifier of the user who created the model.
    - created_at (datetime): Timestamp of when the model was created.
    - updated_at (datetime): Timestamp of when the model was last updated.
    - author (User): Relationship to the user who created the model.

    Methods:
    None
    """

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255), nullable=True)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    author = relationship("User")


class ModelVersion(Base):
    """
    Represents a version of a machine learning model in the application.

    Attributes:
    - id (int): Unique identifier for the model version.
    - model_id (int): Identifier of the model this version belongs to.
    - version (str): Version number of the model.
    - model_path (str): Path to the saved model file.
    - metrics (JSON): Metrics associated with the model version.
    - created_at (datetime): Timestamp of when the model version was created.
    - model (Model): Relationship to the model this version belongs to.

    Methods:
    None
    """

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    version = Column(String(20), nullable=False)
    model_path = Column(String(200), nullable=False)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    model = relationship("Model")


class Deployment(Base):
    """
    Represents a deployment of a machine learning model version in different environments.

    Attributes:
    - id (int): Unique identifier for the deployment.
    - model_version_id (int): Identifier of the model version being deployed.
    - environment (str): Environment where the model is being deployed (e.g., 'staging', 'production').
    - deployed_at (datetime): Timestamp of when the model was deployed.
    - status (str): Status of the deployment (e.g., 'deployed', 'failed').

    Relationships:
    - model_version (ModelVersion): Relationship to the model version being deployed.

    Methods:
    None
    """

    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey('model_versions.id'), nullable=False)
    environment = Column(String(20), nullable=False) 
    deployed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(String(20), nullable=False)  

    model_version = relationship("ModelVersion")