from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from ..crud.settings import settings
from ..crud.db import get_db
from ..models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


class AuthService:
    """
    A class to handle authentication and authorization functionalities.
    """

    def __init__(self, secret_key: str, algorithm: str, access_token_expire_minutes: int):
        """
        Initialize the authentication service with secret key, algorithm, and access token expiration time.

        :param secret_key: The secret key used for JWT encoding and decoding.
        :param algorithm: The algorithm used for JWT encoding and decoding.
        :param access_token_expire_minutes: The expiration time of access tokens in minutes.
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a new access token using the provided data and expiration delta.

        :param data: The data to be included in the JWT payload.
        :param expires_delta: The optional expiration delta for the token. If not provided, the token will expire after the default access token expiration time.
        :return: The encoded JWT access token.
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str, credentials_exception: HTTPException) -> User:
        """
        Verify the provided JWT token and return the associated user.

        :param token: The JWT token to be verified.
        :param credentials_exception: The HTTPException to be raised if the token is invalid.
        :return: The User object associated with the verified token.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
        return username

    def authenticate_user(self, db: Session, username: str, password: str):
        """
        Authenticate the user by verifying their username and password.

        :param db: The SQLAlchemy database session.
        :param username: The username of the user to be authenticated.
        :param password: The password of the user to be authenticated.
        :return: The authenticated User object if the credentials are valid, otherwise None.
        """
        user = db.query(User).filter(User.username == username).first()
        if user and user.verify_password(password):
            return user
        return None

    def get_current_user(self, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
        """
        Get the current user based on the provided JWT token.

        :param token: The JWT token to be used for user authentication.
        :param db: The SQLAlchemy database session.
        :return: The authenticated User object associated with the provided token.
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        user = self.verify_token(token, credentials_exception)
        return user


auth_service = AuthService(settings.SECRET_KEY, settings.ALGORITHM, settings.ACCESS_TOKEN_EXPIRE_MINUTES)