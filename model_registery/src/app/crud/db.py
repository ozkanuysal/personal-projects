from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .settings import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    This function creates and yields a database session using SQLAlchemy.
    The session is automatically closed after the function's execution, ensuring proper resource management.

    Parameters:
    None

    Returns:
    db (sqlalchemy.orm.session.Session): A database session object.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()