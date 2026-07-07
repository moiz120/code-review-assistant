"""
Test script for the Code Review API.
Sends sample PR data and prints the review response.

Usage:
    # First, start the server:
    # uvicorn app.main:app --reload --port 8000

    # Then run tests:
    python scripts/test_api.py
"""
import requests
import json

API_BASE = "http://localhost:8000/api/v1"


def test_health():
    """Test health endpoint."""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_review():
    """Test review generation endpoint."""
    print("\n=== Testing Review Generation ===")

    # Sample PR payload
    payload = {
        "pr_payload": {
            "pr_number": 42,
            "repository": "my-project",
            "owner": "my-org",
            "title": "Add user authentication feature",
            "description": "Implements JWT-based authentication",
            "branch": "feature/auth",
            "base_branch": "main",
            "author": "developer1",
            "changes": [
                {
                    "filename": "auth.py",
                    "status": "added",
                    "additions": 45,
                    "deletions": 0,
                    "patch": "+import os\n+import jwt\n+from datetime import datetime, timedelta\n+\n+SECRET_KEY = 'my-secret-key-123'\n+\n+def create_token(user_id):\n+    payload = {\n+        'user_id': user_id,\n+        'exp': datetime.utcnow() + timedelta(hours=24)\n+    }\n+    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')\n+\n+def verify_token(token):\n+    try:\n+        return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n+    except:\n+        return None",
                    "new_content": "import os\nimport jwt\nfrom datetime import datetime, timedelta\n\nSECRET_KEY = 'my-secret-key-123'\n\ndef create_token(user_id):\n    payload = {\n        'user_id': user_id,\n        'exp': datetime.utcnow() + timedelta(hours=24)\n    }\n    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')\n\ndef verify_token(token):\n    try:\n        return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n    except:\n        return None"
                },
                {
                    "filename": "app.py",
                    "status": "modified",
                    "additions": 20,
                    "deletions": 5,
                    "patch": "+def process_data(data):\n+    result = []\n+    for item in data:\n+        result.append(item * 2)\n+    return result",
                    "new_content": "def process_data(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result"
                }
            ],
            "installation_id": None
        },
        "review_style": "balanced",
        "focus_areas": None
    }

    response = requests.post(f"{API_BASE}/review", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nReview ID: {data['review_id']}")
        print(f"Overall Score: {data['overall_score']}/100")
        print(f"Processing Time: {data['processing_time_ms']}ms")
        print(f"\nSummary:\n{data['summary']}")
        print(f"\nComments ({len(data['comments'])}):")
        for comment in data['comments']:
            print(f"\n  [{comment['severity'].upper()}] {comment['category']}")
            print(f"   File: {comment['file_path']}:{comment.get('line_number', 'N/A')}")
            print(f"   {comment['message'][:200]}")
            if comment.get('suggestion'):
                print(f"   Suggestion: {comment['suggestion'][:150]}...")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def test_rag_stats():
    """Test RAG statistics endpoint."""
    print("\n=== Testing RAG Stats ===")
    response = requests.get(f"{API_BASE}/rag/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_model_load():
    """Test model loading endpoint."""
    print("\n=== Testing Model Load ===")
    print("Loading model (this may take a few minutes)...")
    response = requests.post(f"{API_BASE}/model/load", params={"use_finetuned": False})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def main():
    print("=" * 60)
    print("Code Review API Test Suite")
    print("=" * 60)

    # Test health first
    if not test_health():
        print("\nServer not running! Start it with: uvicorn app.main:app --reload")
        return

    # Test RAG stats
    test_rag_stats()

    # Test review without model (AST analysis only)
    print("\n--- Testing without LLM (AST analysis only) ---")
    test_review()

    # Load model and test with LLM
    print("\n--- Testing with LLM ---")
    if test_model_load():
        test_review()

    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
