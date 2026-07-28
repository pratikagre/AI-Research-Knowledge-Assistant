import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from src.database.base import engine, Base
from routes import document_routes, search_routes, analysis_routes, analytics_routes

# Initialize SQLite database tables on startup
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="A production-grade assistant for ingestion, hybrid retrieval, ML classification, and grounded RAG on PDF repositories.",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(document_routes.router)
app.include_router(search_routes.router)
app.include_router(analysis_routes.router)
app.include_router(analytics_routes.router)

@app.get("/", tags=["General"])
async def root():
    """
    Root API endpoint.
    """
    return {
        "app_name": settings.APP_NAME,
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "online"
    }

if __name__ == "__main__":
    uvicorn.run("main.py:app", host="127.0.0.1", port=8000, reload=True)
