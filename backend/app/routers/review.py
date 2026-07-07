"""
FastAPI router for code review endpoints.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List
from app.models.schemas import ReviewRequest, ReviewResponse, HealthCheck
from app.services.review_orchestrator import review_orchestrator
from app.services.llm_service import llm_service
from app.services.rag_engine import rag_engine
from app.core.logging import logger
import time

router = APIRouter(prefix="/api/v1", tags=["code-review"])

# Track server start time for uptime
START_TIME = time.time()


@router.post("/review", response_model=ReviewResponse)
async def create_review(request: ReviewRequest):
    """
    Generate a code review for a pull request.

    This endpoint processes a PR payload, analyzes code changes,
    retrieves relevant historical context, and generates structured review comments.
    """
    logger.info("review_request_received", 
               pr_number=request.pr_payload.pr_number,
               repo=request.pr_payload.repository)

    try:
        response = await review_orchestrator.process_review(request)
        return response
    except Exception as e:
        logger.error("review_generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Review generation failed: {str(e)}")


@router.post("/review/github-webhook")
async def github_webhook(payload: dict, background_tasks: BackgroundTasks):
    """
    Receive GitHub webhook events for PR reviews.

    Expected payload format from GitHub PR webhook.
    """
    logger.info("github_webhook_received", event_type=payload.get("action"))

    # Parse GitHub webhook payload
    try:
        if payload.get("action") in ["opened", "synchronize", "reopened"]:
            # Extract PR info
            pr_data = payload.get("pull_request", {})

            # Build changes list from files
            # Note: In production, you'd fetch files via GitHub API
            changes = []

            review_request = ReviewRequest(
                pr_payload={
                    "pr_number": pr_data.get("number"),
                    "repository": payload.get("repository", {}).get("name"),
                    "owner": payload.get("repository", {}).get("owner", {}).get("login"),
                    "title": pr_data.get("title"),
                    "description": pr_data.get("body"),
                    "branch": pr_data.get("head", {}).get("ref"),
                    "base_branch": pr_data.get("base", {}).get("ref"),
                    "author": pr_data.get("user", {}).get("login"),
                    "changes": changes,
                    "installation_id": payload.get("installation", {}).get("id")
                }
            )

            # Process review in background
            background_tasks.add_task(review_orchestrator.process_review, review_request)

            return {"status": "processing", "message": "Review queued for processing"}

        return {"status": "ignored", "message": "Event type not handled"}

    except Exception as e:
        logger.error("webhook_processing_failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Invalid webhook payload: {str(e)}")


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return HealthCheck(
        status="healthy",
        model_loaded=llm_service.loaded,
        vector_store_ready=rag_engine.index is not None,
        uptime_seconds=int(time.time() - START_TIME)
    )


@router.get("/rag/stats")
async def get_rag_stats():
    """
    Get statistics about the RAG vector store.
    """
    return rag_engine.get_stats()


@router.post("/model/load")
async def load_model(use_finetuned: bool = True):
    """
    Load the LLM model into memory.
    Call this before generating reviews if model is not loaded.
    """
    try:
        result = llm_service.load_model(use_finetuned=use_finetuned)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


@router.post("/model/unload")
async def unload_model():
    """
    Unload the LLM model to free memory.
    """
    llm_service.unload_model()
    return {"status": "success", "message": "Model unloaded"}