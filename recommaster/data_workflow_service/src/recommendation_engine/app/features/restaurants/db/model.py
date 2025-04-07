import uuid
import pendulum
from sqlalchemy import Column
from clickhouse_sqlalchemy import engines
from clickhouse_sqlalchemy.types import String, Float, Int32, Nullable

from ....shared_kernel.database.clickhouse import ClickhouseBase


class RestaurantModel(ClickhouseBase):
    __tablename__ = "restaurant"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    provider = Column(String)
    name = Column(String)
    rating = Column(Nullable(Float))
    restaurant_id = Column(String)
    delivery_fee = Column(Nullable(Float))
    restaurant_slug = Column(String)
    delivery_time = Column(Nullable(String))
    delivery_fee_currency = Column(Nullable(String))
    review_number = Column(Int32, default=0)
    image_url = Column(Nullable(String))
    order_amount = Column(Nullable(Float))
    order_amount_currency = Column(Nullable(String))
    loyalty_percentage_amount = Column(Nullable(Float))
    lat = Column(Float)
    lon = Column(Float)
    city = Column(String)
    version = Column(Int32, default=pendulum.now("Europe/Istanbul").int_timestamp)

    __table_args__ = (
        engines.ReplacingMergeTree(order_by=["id"], version="version"),
        {"schema": "default"},
    )