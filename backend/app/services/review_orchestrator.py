"""
Review Orchestrator - Main service that coordinates all components.
Combines AST analysis, RAG retrieval, and LLM generation into a unified pipeline.
"""
import uuid
import time
from typing import List, Optional
from app.models.schemas import (
    ReviewRequest, ReviewResponse, ReviewComment,
    CodeChange, PullRequestPayload
)
from app.services.code_analyzer import code_analyzer
from app.services.rag_engine import rag_engine
from app.services.llm_service import llm_service
from app.core.logging import logger


class ReviewOrchestrator:
    """
    Orchestrates the complete code review pipeline:
    1. Parse PR payload
    2. Analyze each file with AST
    3. Retrieve RAG context
    4. Generate LLM review
    5. Compile and return structured response
    """

    def __init__(self):
        self.logger = logger.bind(service="review_orchestrator")

    async def process_review(self, request: ReviewRequest) -> ReviewResponse:
        """
        Process a complete code review request.

        Args:
            request: ReviewRequest with PR payload and preferences

        Returns:
            ReviewResponse with all generated comments
        """
        start_time = time.time()
        review_id = str(uuid.uuid4())[:8]

        self.logger.info("starting_review", 
                        review_id=review_id,
                        pr_number=request.pr_payload.pr_number,
                        repo=request.pr_payload.repository)

        all_comments: List[ReviewComment] = []

        # Process each file change
        for change in request.pr_payload.changes:
            self.logger.info("processing_file", 
                           review_id=review_id,
                           file=change.filename)

            try:
                # Step 1: AST Analysis
                ast_results = code_analyzer.analyze_change(change)

                # Convert AST issues to ReviewComments
                for issue in ast_results.get("ast_issues", []):
                    comment = ReviewComment(
                        id=f"{review_id}-ast-{len(all_comments)+1:03d}",
                        category=issue["category"],
                        severity=issue["severity"],
                        file_path=change.filename,
                        line_number=issue.get("line"),
                        message=issue["message"],
                        confidence_score=0.95,
                        model_used="ast-analyzer"
                    )
                    all_comments.append(comment)

                for issue in ast_results.get("security_flags", []):
                    comment = ReviewComment(
                        id=f"{review_id}-sec-{len(all_comments)+1:03d}",
                        category=issue["category"],
                        severity=issue["severity"],
                        file_path=change.filename,
                        line_number=issue.get("line"),
                        message=issue["message"],
                        confidence_score=0.90,
                        model_used="security-scanner"
                    )
                    all_comments.append(comment)

                for issue in ast_results.get("style_flags", []):
                    comment = ReviewComment(
                        id=f"{review_id}-sty-{len(all_comments)+1:03d}",
                        category=issue["category"],
                        severity=issue["severity"],
                        file_path=change.filename,
                        line_number=issue.get("line"),
                        message=issue["message"],
                        confidence_score=0.80,
                        model_used="style-checker"
                    )
                    all_comments.append(comment)

                # Step 2: RAG Retrieval for context
                rag_context = ""
                if rag_engine.index is not None and change.new_content:
                    rag_context = rag_engine.retrieve_similar(change.new_content)

                # Step 3: LLM Review (if model is loaded)
                if llm_service.loaded:
                    llm_comments = llm_service.generate_review(
                        change, 
                        ast_results.get("ast_issues", []),
                        rag_context=rag_context
                    )

                    # Add review_id prefix to LLM comments
                    for comment in llm_comments:
                        comment.id = f"{review_id}-llm-{len(all_comments)+1:03d}"

                    all_comments.extend(llm_comments)

            except Exception as e:
                self.logger.error("file_processing_error",
                                review_id=review_id,
                                file=change.filename,
                                error=str(e))
                continue

        # Calculate overall score (100 - penalty per issue)
        score = self._calculate_score(all_comments)

        # Generate summary from comments
        summary = self._generate_summary(all_comments, request.pr_payload.title)

        processing_time = int((time.time() - start_time) * 1000)

        self.logger.info("review_complete",
                        review_id=review_id,
                        total_comments=len(all_comments),
                        processing_time_ms=processing_time)

        return ReviewResponse(
            review_id=review_id,
            pr_number=request.pr_payload.pr_number,
            repository=request.pr_payload.repository,
            summary=summary,
            comments=all_comments,
            overall_score=score,
            processing_time_ms=processing_time,
            model_version="groq-llama3-8b"
        )

    def _calculate_score(self, comments: List[ReviewComment]) -> float:
        """
        Calculate overall code quality score.

        Args:
            comments: All review comments

        Returns:
            Score from 0-100
        """
        if not comments:
            return 100.0

        # Penalty weights by severity
        penalties = {
            "critical": 15,
            "high": 8,
            "medium": 4,
            "low": 1,
            "info": 0
        }

        total_penalty = sum(
            penalties.get(c.severity.value, 0) 
            for c in comments
        )

        score = max(0, 100 - total_penalty)
        return round(score, 1)

    def _generate_summary(self, comments: List[ReviewComment], title: str) -> str:
        """Generate a summary from comments."""
        if not comments:
            return "No issues found. Great job!"
        
        critical = sum(1 for c in comments if c.severity.value == "critical")
        high = sum(1 for c in comments if c.severity.value == "high")
        medium = sum(1 for c in comments if c.severity.value == "medium")
        low = sum(1 for c in comments if c.severity.value == "low")
        
        parts = []
        if critical > 0:
            parts.append(f"{critical} critical")
        if high > 0:
            parts.append(f"{high} high")
        if medium > 0:
            parts.append(f"{medium} medium")
        if low > 0:
            parts.append(f"{low} low")
        
        return f"Found {len(comments)} issues ({', '.join(parts)}) in PR: {title}"


# Singleton instance
review_orchestrator = ReviewOrchestrator()