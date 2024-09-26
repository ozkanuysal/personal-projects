from typing import Any, Callable
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI, status
from sqlalchemy.exc import SQLAlchemyError

class BooklyException(Exception):
    """This is the base class for all bookly errors"""

    pass


class InvalidToken(BooklyException):
    """User has provided an invalid or expired token"""

    pass


class RevokedToken(BooklyException):
    """User has provided a token that has been revoked"""

    pass


class AccessTokenRequired(BooklyException):
    """User has provided a refresh token when an access token is needed"""

    pass


class RefreshTokenRequired(BooklyException):
    """User has provided an access token when a refresh token is needed"""

    pass


class UserAlreadyExists(BooklyException):
    """User has provided an email for a user who exists during sign up."""

    pass


class InvalidCredentials(BooklyException):
    """User has provided wrong email or password during log in."""

    pass


class InsufficientPermission(BooklyException):
    """User does not have the neccessary permissions to perform an action."""

    pass


class BookNotFound(BooklyException):
    """Book Not found"""

    pass


class TagNotFound(BooklyException):
    """Tag Not found"""

    pass


class TagAlreadyExists(BooklyException):
    """Tag already exists"""

    pass


class UserNotFound(BooklyException):
    """User Not found"""

    pass


class AccountNotVerified(Exception):
    """Account not yet verified"""
    pass