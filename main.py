from fastapi import FastAPI
from app.db.database import engine, Base
from app.db import models
from app.api.analysis_endpoints import router as analysis_router

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for cleaning text and performing sentiment analysis.",
    version="0.1.0"
)

app.include_router(analysis_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def reat_root():
    return {"message": "Welcome to the Sentiment Analyzer API v0.1"}