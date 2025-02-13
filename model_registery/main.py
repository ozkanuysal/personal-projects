from prometheus_client import CollectorRegistry
from prometheus_client import generate_latest
from sqlalchemy_utils import database_exists, create_database
from src.app.crud.celery_app import celery_app
from src.app.api import *
from src.app.crud.app import app, Base
from src.app.crud.db import SessionLocal, engine
from src.app.services.data_ingestion import data_ingestion_service
from fastapi.responses import Response


Base.metadata.create_all(bind=engine)

app.include_router(auth_router)
app.include_router(health_check_router)
app.include_router(model_router)
app.include_router(inference_router)



@app.get("/metrics")
def metrics():
    registry = CollectorRegistry()
    data = generate_latest(registry).decode("utf-8")
    return Response(content=data, media_type="text/plain; version=0.0.4", headers={"Content-Encoding": "identity"})


@app.on_event("startup")
def on_startup():
    if not database_exists(engine.url):
        create_database(engine.url)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    data_ingestion_service.create_initial_data(db)


@app.on_event("shutdown")
def on_shutdown():
    db = SessionLocal()
    db.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)