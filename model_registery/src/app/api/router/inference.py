from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...crud.db import get_db
from ...crud.logger import AppLogger
from ...models.user import User
from ...services.auth import auth_service
from ...services.inference import inference_service
from ...services.model_registry import model_registry_service

logger = AppLogger(__name__).get_logger()

inference_router = APIRouter(tags=["Inference"])


@inference_router.get("/inference")
async def get_model(
        model_name: str,
        model_version: str,
        customer_id: int,
        purchase_date_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(auth_service.get_current_user)
):
    """
    This function handles the inference request for a specific model and version.
    It logs the request, fetches the model and its artifacts, performs inference,
    and logs the result. If an error occurs during the process, it logs the error
    and raises an HTTPException.

    Parameters:
    - model_name (str): The name of the model.
    - model_version (str): The version of the model.
    - customer_id (int): The ID of the customer for whom the inference is requested.
    - purchase_date_id (str): The ID of the purchase date for which the inference is requested.
    - db (Session): The database session (optional, default: Depends(get_db)).
    - current_user (User): The current user (optional, default: Depends(auth_service.get_current_user)).

    Returns:
    dict: A dictionary containing the prediction result.

    Raises:
    HTTPException: If an error occurs during the inference process.
    """
    try:
        logger.info("User %s requested inference for model: %s, version: %s, customer_id: %d, purchase_date_id: %s",
                    current_user, model_name, model_version, customer_id, purchase_date_id)

        # TODO: Model and artifact file extensions may differ. It should be able to load more than one file type automatically.
        # TODO: There is only config.json artifact in the developed model. More than one artifact should be able to be handled.
        model_version_record, model_file_path = model_registry_service.get_model(model_name, model_version)
        artifact_path = model_registry_service.get_artifacts(model_name, model_version)[0]

        params = {
            "customer_id": customer_id,
            "purchase_date_id": purchase_date_id
        }
        res = inference_service.test_inference(params, model_file_path, artifact_path)
        logger.info("Inference completed successfully for user: %s, model: %s, version: %s",
                    current_user, model_name, model_version)
        return {"prediction": res}
    except Exception as e:
        logger.error(
            "Error during inference for model: %s, version: %s, customer_id: %d, purchase_date_id: %s for user: %s. Error: %s",
            model_name, model_version, customer_id, purchase_date_id, current_user, str(e))
        raise HTTPException(status_code=500, detail=str(e))