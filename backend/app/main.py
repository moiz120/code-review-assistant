"""
Code Review Assistant - FastAPI Application
Main entry point for the API server.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import review
from app.core.logging import logger
from app.services.llm_service import llm_service
from app.services.rag_engine import rag_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("application_starting")

    # Initialize RAG engine (loads vector store)
    logger.info("rag_engine_initialized", stats=rag_engine.get_stats())

    # Note: LLM model is NOT loaded automatically on startup
    # to save memory. Call POST /api/v1/model/load when ready.
    logger.info("ready_to_serve")

    yield

    # Shutdown
    logger.info("application_shutting_down")

    # Save RAG index
    rag_engine.save_index()

    # Unload model if loaded
    if llm_service.model_loaded:
        llm_service.unload_model()

    logger.info("shutdown_complete")


# Create FastAPI application
app = FastAPI(
    title="Code Review Assistant",
    description="AI-powered code review system with RAG and fine-tuned LLMs",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(review.router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Code Review Assistant",
        "version": "1.0.0",
        "description": "AI-powered code review with RAG and fine-tuned LLMs",
        "endpoints": {
            "review": "POST /api/v1/review",
            "health": "GET /api/v1/health",
            "model_load": "POST /api/v1/model/load",
            "rag_stats": "GET /api/v1/rag/stats"
        }
    }
