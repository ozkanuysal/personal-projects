from passlib.context import CryptContext
from sqlalchemy.ext.declarative import declarative_base
from fastapi import FastAPI


Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="Model Registery", version="0.1.0")