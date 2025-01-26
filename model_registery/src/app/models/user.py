from sqlalchemy import Column, Integer, String

from ..crud.app import Base, pwd_context


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(20), nullable=False)  # e.g., 'admin', 'developer'
    password_hash = Column(String(255), nullable=False)

    def verify_password(self, password: str) -> bool:
        """
        Verifies if the provided password matches the hashed password.

        Parameters:
        password (str): The password to be verified.

        Returns:
        bool: True if the password matches the hashed password, False otherwise.
        """
        return pwd_context.verify(password, self.password_hash)

    def set_password(self, password: str):
        """
        Sets the password by hashing it using the provided context.

        Parameters:
        password (str): The password to be hashed.

        Returns:
        None
        """
        self.password_hash = pwd_context.hash(password)