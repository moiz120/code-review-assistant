"""LLM Service for code review using Groq API."""
import os
import json
import re
from typing import List, Optional
from groq import Groq
from app.models.schemas import CodeChange, ReviewComment, ReviewCategory, Severity
from app.core.config import settings
from app.core.logging import logger


class LLMService:
    """Uses Groq API for fast, high-quality code review generation."""
    
    def __init__(self):
        self.logger = logger.bind(service="llm_service")
        self.client: Optional[Groq] = None
        self.model_name = "llama3-8b-8192"
        self.loaded = False
    
    def load_model(self, use_finetuned: bool = False) -> dict:
        """Initialize Groq client (no heavy model download)."""
        try:
            api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
            if not api_key or api_key == "gsk_your_actual_key_here":
                return {
                    "status": "error",
                    "message": "GROQ_API_KEY not set. Add it to your .env file."
                }
            
            self.client = Groq(api_key=api_key)
            self.loaded = True
            self.logger.info("groq_client_initialized", model=self.model_name)
            return {
                "status": "success",
                "message": "Groq API client initialized successfully"
            }
        except Exception as e:
            self.logger.error("groq_init_failed", error=str(e))
            return {
                "status": "error",
                "message": "Failed to initialize Groq: " + str(e)
            }
    
    def generate_review(self, change: CodeChange, ast_issues: List[dict] = None, 
                       rag_context: str = "") -> List[ReviewComment]:
        """Generate AI review using Groq API."""
        if not self.loaded or not self.client:
            return []
        
        code = change.new_content or ""
        if not code:
            return []
        
        # Build AST summary
        ast_summary = ""
        if ast_issues:
            parts = ["AST detected issues:"]
            for i in ast_issues[:5]:
                sev = i.get("severity", "UNKNOWN")
                msg = i.get("message", "")
                parts.append("- [" + sev + "] " + msg)
            ast_summary = "\n".join(parts)
        
        # Build RAG context section
        rag_section = ""
        if rag_context:
            rag_section = "\nSimilar past issues:\n" + rag_context
        
        # Build prompt (Python 3.14 safe - no triple-quoted f-strings)
        prompt_lines = [
            "You are an expert code reviewer. Review this code and identify issues.",
            "",
            "File: " + change.filename,
            "",
            "Code:",
            "```python",
            code,
            "```",
        ]
        
        if ast_summary:
            prompt_lines.append("")
            prompt_lines.append(ast_summary)
        
        if rag_section:
            prompt_lines.append("")
            prompt_lines.append(rag_section)
        
        prompt_lines.extend([
            "",
            "Respond with a JSON array of issues. Each issue must have:",
            '- "severity": "CRITICAL", "HIGH", "MEDIUM", or "LOW"',
            '- "category": "SECURITY", "FUNCTIONAL", "PERFORMANCE", "STYLE", or "DOCUMENTATION"',
            '- "message": detailed explanation with fix suggestion',
            "",
            "If no issues found, return an empty array []."
        ])
        
        prompt = "\n".join(prompt_lines)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON array from response
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                issues = json.loads(json_match.group())
            else:
                issues = self._parse_text_response(content)
            
            comments = []
            for issue in issues:
                sev_val = issue.get("severity", "LOW").lower()
                cat_val = issue.get("category", "FUNCTIONAL").lower()
                comments.append(ReviewComment(
                    severity=Severity(sev_val),
                    message=issue.get("message", ""),
                    category=ReviewCategory(cat_val),
                    line_number=issue.get("line_number"),
                    file_path=change.filename
                ))
            
            self.logger.info("llm_review_complete", comments_generated=len(comments))
            return comments
            
        except Exception as e:
            self.logger.error("llm_generation_failed", error=str(e))
            return []
    
    def _parse_text_response(self, text: str) -> List[dict]:
        """Fallback parser for non-JSON responses."""
        issues = []
        lines = text.split("\n")
        current = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("SEVERITY:") or line.startswith("Severity:"):
                if current:
                    issues.append(current)
                current = {"severity": line.split(":", 1)[1].strip().upper()}
            elif line.startswith("CATEGORY:") or line.startswith("Category:"):
                current["category"] = line.split(":", 1)[1].strip().upper()
            elif line.startswith("MESSAGE:") or line.startswith("Message:") or line.startswith("-"):
                current["message"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif line and current.get("message"):
                current["message"] += " " + line
        
        if current:
            issues.append(current)
        
        return issues
    
    def unload_model(self) -> dict:
        """Unload (just clears client reference)."""
        self.client = None
        self.loaded = False
        self.logger.info("groq_client_unloaded")
        return {"status": "success", "message": "Groq client unloaded"}


# Singleton instance
llm_service = LLMService()