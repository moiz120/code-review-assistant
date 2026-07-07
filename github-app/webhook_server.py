"""
GitHub App Webhook Server.
Receives GitHub webhook events and triggers code reviews.

This is a standalone server that can be deployed separately from the main API.
It handles GitHub App authentication and webhook verification.
"""
import os
import json
import hmac
import hashlib
import base64
from typing import Optional
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

app = FastAPI(title="GitHub App Webhook Handler")

# Configuration
APP_ID = os.getenv("GITHUB_APP_ID", "")
PRIVATE_KEY_PATH = os.getenv("GITHUB_PRIVATE_KEY_PATH", "./github-app-private-key.pem")
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


def load_private_key():
    """Load GitHub App private key from PEM file."""
    with open(PRIVATE_KEY_PATH, "r") as f:
        private_key = f.read()

    return serialization.load_pem_private_key(
        private_key.encode(),
        password=None,
        backend=default_backend()
    )


def generate_jwt() -> str:
    """Generate JWT for GitHub App authentication."""
    private_key = load_private_key()

    payload = {
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc).replace(hour=datetime.now(timezone.utc).hour + 10),
        "iss": APP_ID
    }

    return jwt.encode(payload, private_key, algorithm="RS256")


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature."""
    if not WEBHOOK_SECRET:
        return True  # Skip verification if no secret configured

    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


def get_installation_token(installation_id: int) -> str:
    """Get installation access token for API calls."""
    jwt_token = generate_jwt()

    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = httpx.post(url, headers=headers)
    response.raise_for_status()

    return response.json()["token"]


async def fetch_pr_files(owner: str, repo: str, pr_number: int, token: str):
    """Fetch files changed in a PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def fetch_file_content(owner: str, repo: str, path: str, ref: str, token: str):
    """Fetch file content at specific ref."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {"ref": ref}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        if response.status_code == 200:
            content = response.json()
            if content.get("encoding") == "base64":
                return base64.b64decode(content["content"]).decode("utf-8")
        return None


async def post_review_comment(owner: str, repo: str, pr_number: int, 
                             token: str, body: str):
    """Post a review comment on a PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"body": body}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()


async def process_pr_review(payload: dict):
    """Process a PR review request."""
    pr_data = payload.get("pull_request", {})
    repo_data = payload.get("repository", {})
    installation_id = payload.get("installation", {}).get("id")

    owner = repo_data.get("owner", {}).get("login")
    repo = repo_data.get("name")
    pr_number = pr_data.get("number")

    print(f"Processing PR #{pr_number} in {owner}/{repo}")

    try:
        # Get installation token
        token = get_installation_token(installation_id)

        # Fetch PR files
        files = await fetch_pr_files(owner, repo, pr_number, token)

        # Build changes list
        changes = []
        for file_data in files:
            if file_data.get("status") == "removed":
                continue

            # Fetch current file content
            new_content = await fetch_file_content(
                owner, repo, file_data["filename"], 
                pr_data["head"]["sha"], token
            )

            changes.append({
                "filename": file_data["filename"],
                "status": file_data.get("status"),
                "additions": file_data.get("additions", 0),
                "deletions": file_data.get("deletions", 0),
                "patch": file_data.get("patch", ""),
                "new_content": new_content
            })

        # Build review request
        review_request = {
            "pr_payload": {
                "pr_number": pr_number,
                "repository": repo,
                "owner": owner,
                "title": pr_data.get("title"),
                "description": pr_data.get("body"),
                "branch": pr_data.get("head", {}).get("ref"),
                "base_branch": pr_data.get("base", {}).get("ref"),
                "author": pr_data.get("user", {}).get("login"),
                "changes": changes,
                "installation_id": installation_id
            },
            "review_style": "balanced",
            "focus_areas": None
        }

        # Send to main API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/review",
                json=review_request,
                timeout=300.0
            )
            review_data = response.json()

        # Format and post review
        review_body = format_review_for_github(review_data)
        await post_review_comment(owner, repo, pr_number, token, review_body)

        print(f"Review posted for PR #{pr_number}")

    except Exception as e:
        print(f"Error processing PR #{pr_number}: {str(e)}")


def format_review_for_github(review_data: dict) -> str:
    """Format review data into a GitHub comment."""
    lines = [
        "## 🤖 AI Code Review",
        "",
        f"**Overall Score:** {review_data.get('overall_score', 0)}/100",
        f"**Processing Time:** {review_data.get('processing_time_ms', 0)}ms",
        "",
        "---",
        ""
    ]

    # Add summary
    summary = review_data.get("summary", "")
    if summary:
        lines.append(summary)
        lines.append("")

    # Add comments
    comments = review_data.get("comments", [])
    if comments:
        lines.append("### Detailed Comments")
        lines.append("")

        # Group by severity
        severity_order = ["critical", "high", "medium", "low", "info"]
        severity_emoji = {
            "critical": "🔴",
            "high": "🟠", 
            "medium": "🟡",
            "low": "🔵",
            "info": "ℹ️"
        }

        for sev in severity_order:
            sev_comments = [c for c in comments if c.get("severity") == sev]
            if sev_comments:
                lines.append(f"\n#### {severity_emoji.get(sev, '•')} {sev.upper()} ({len(sev_comments)})")
                for comment in sev_comments:
                    lines.append(f"\n**{comment.get('file_path', 'Unknown')}:{comment.get('line_number', 'N/A')}**")
                    lines.append(f"- Category: {comment.get('category', 'unknown')}")
                    lines.append(f"- {comment.get('message', '')}")
                    if comment.get('suggestion'):
                        lines.append(f"- 💡 Suggestion: {comment['suggestion']}")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by Code Review Assistant*")

    return "\n".join(lines)


@app.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None)
):
    """Handle GitHub webhook events."""
    # Read payload
    payload = await request.body()

    # Verify signature
    if x_hub_signature_256 and not verify_signature(payload, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse JSON
    data = json.loads(payload)

    # Handle PR events
    if x_github_event == "pull_request":
        action = data.get("action")

        if action in ["opened", "synchronize", "reopened"]:
            # Process in background
            background_tasks.add_task(process_pr_review, data)
            return JSONResponse(
                content={"status": "processing", "message": "Review queued"},
                status_code=202
            )

        return JSONResponse(
            content={"status": "ignored", "message": f"Action '{action}' not handled"},
            status_code=200
        )

    return JSONResponse(
        content={"status": "ignored", "message": f"Event '{x_github_event}' not handled"},
        status_code=200
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "app_id": APP_ID}
