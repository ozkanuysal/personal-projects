from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette import status

from ...crud.db import get_db
from ...crud.logger import AppLogger
from ...crud.settings import settings
from ...services.auth import auth_service

logger = AppLogger(__name__).get_logger()

auth_router = APIRouter()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class TokenRequest(BaseModel):
    username: str
    password: str


@auth_router.post("/login", response_model=TokenResponse)
def login_for_access_token(db: Session = Depends(get_db), form_data: TokenRequest = Depends()):
    """
    Authenticates a user and generates an access token for subsequent API requests.

    Parameters:
    db (Session): A database session object provided by the FastAPI Depends mechanism.
    form_data (TokenRequest): A Pydantic model containing the username and password.

    Returns:
    dict: A dictionary containing the access token and token type.

    Raises:
    HTTPException: If the username or password is incorrect, a 401 Unauthorized HTTP exception is raised.
    """
    logger.debug("Attempting to authenticate user with username: %s", form_data.username)
    user = auth_service.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.debug("User authenticated successfully: %s", form_data.username)
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}