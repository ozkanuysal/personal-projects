from sqlalchemy import Column, Integer, Date, Float

from ..crud.app import Base


class Customer(Base):
    """
    A class representing a customer in the database.

    Attributes
    ----------
    __tablename__ : str
        The name of the table in the database.
    customer_id : Column(Integer, primary_key=True)
        The unique identifier for each customer.
    age : Column(Integer, nullable=True)
        The age of the customer. Can be null.
    gender : Column(Integer, nullable=True)
        The gender of the customer. Can be null.
    annual_income : Column(Float, nullable=True)
        The annual income of the customer. Can be null.
    """
    __tablename__ = 'customer'
    customer_id = Column(Integer, primary_key=True)
    age = Column(Integer, nullable=True)
    gender = Column(Integer, nullable=True)
    annual_income = Column(Float, nullable=True)


class Transaction(Base):
    """
    A class representing a transaction in the database.

    Attributes
    ----------
    __tablename__ : str
        The name of the table in the database.
    id : Column(Integer, primary_key=True, autoincrement=True)
        The unique identifier for each transaction.
    customer_id : Column(Integer)
        The identifier of the customer associated with the transaction.
    purchase_amount : Column(Float)
        The amount of the transaction.
    purchase_date : Column(Date)
        The date of the transaction.
    """
    __tablename__ = 'transaction'
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer)
    purchase_amount = Column(Float)
    purchase_date = Column(Date)