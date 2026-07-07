# Code Review Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Groq-FF6B6B?style=flat&logo=openai&logoColor=white" alt="Groq">
  <img src="https://img.shields.io/badge/FAISS-2563EB?style=flat&logo=meta&logoColor=white" alt="FAISS">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <b>AI-Powered Code Review System with AST Analysis, RAG, and LLM Integration</b>
</p>

<p align="center">
  <img src="assets/dashboard-preview.png" alt="Dashboard Preview" width="800">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

**Code Review Assistant** is an industry-level code review system that combines:

- **Static Analysis (AST)** — Detects security vulnerabilities, anti-patterns, and style issues
- **Retrieval-Augmented Generation (RAG)** — Learns from historical reviews using vector search
- **Large Language Models (LLM)** — Generates human-like review comments with fix suggestions

Built for production environments, it supports GitHub webhooks, multi-file PR reviews, and real-time feedback.

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **AST Analysis** | Detects hardcoded secrets, SQL injection, command injection, bare except clauses, mutable defaults |
| **RAG Engine** | FAISS-powered vector search retrieves similar past reviews for context-aware suggestions |
| **LLM Integration** | Groq API (Llama 3 8B) generates detailed review comments with severity and fix suggestions |
| **Quality Scoring** | Calculates 0-100 code quality score based on issue severity weights |
| **Multi-Style Reviews** | Strict, Balanced, and Lenient review modes |
| **GitHub Webhooks** | Auto-reviews PRs on open, synchronize, and reopen events |
| **REST API** | Fast, async endpoints with OpenAPI documentation |

### Security Detection

- Hardcoded credentials and API keys
- Dangerous `os.system()` and `subprocess` calls
- SQL injection vulnerabilities
- Unsafe deserialization (`pickle`, `eval`, `exec`)
- Weak hashing algorithms (MD5)
- Debug mode enabled in production

### Functional Detection

- Bare `except:` clauses
- Mutable default arguments
- Division by zero risks
- Missing error handling
- Unused imports and variables

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GitHub PR     │────▶│  FastAPI        │────▶│  AST Analyzer   │
│   Webhook       │     │  Backend        │     │  (Security +  │
│                 │     │                 │     │   Functional)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  RAG Engine     │
                       │  (FAISS +       │
                       │   Embeddings)   │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Groq LLM       │
                       │  (Llama 3 8B)   │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Review Response│
                       │  (Score +       │
                       │   Comments)     │
                       └─────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **AI/ML** | Groq API (Llama 3 8B), Sentence Transformers, FAISS |
| **Code Analysis** | Python AST, Security pattern matching |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Data** | FAISS Vector Store, JSON schemas |
| **DevOps** | Pydantic, python-dotenv, structlog |

---

## Installation

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com/keys) (free tier: 1M tokens/day)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/code-review-assistant.git
cd code-review-assistant
```

### Step 2: Set Up Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment

Create `backend/.env`:

```env
# Model Configuration
BASE_MODEL_NAME=groq-llama3-8b
GROQ_API_KEY=gsk_your_groq_api_key_here

# RAG Configuration
VECTOR_STORE_PATH=./data/vector_store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_RETRIEVAL=3

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Hugging Face Token (for embeddings)
HF_TOKEN=your_hf_token_here
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com/keys)

### Step 4: Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

Server will start at `http://localhost:8000`

### Step 5: Open the Dashboard

Open `frontend/index.html` in your browser (double-click the file).

> **Note:** For CORS to work properly, you may need to serve the frontend via a local server:
> ```bash
> cd frontend
> python -m http.server 3000
> # Then open http://localhost:3000
> ```

---

## Usage

### Web Dashboard

1. Open `frontend/index.html` in your browser
2. Paste Python code in the **Source Code** field
3. Select **Review Style** (Strict / Balanced / Lenient)
4. Click **Analyze Code**
5. View score, issue breakdown, and detailed comments

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

#### Generate Review
```bash
curl -X POST http://localhost:8000/api/v1/review \
  -H "Content-Type: application/json" \
  -d '{
    "pr_payload": {
      "pr_number": 1,
      "repository": "my-repo",
      "owner": "my-org",
      "title": "Add authentication",
      "branch": "feature/auth",
      "base_branch": "main",
      "author": "developer",
      "changes": [{
        "filename": "auth.py",
        "status": "added",
        "additions": 50,
        "new_content": "password = \"secret123\"\nimport os\nos.system(\"rm -rf /\")"
      }]
    },
    "review_style": "strict"
  }'
```

#### Load Model
```bash
curl -X POST http://localhost:8000/api/v1/model/load \
  -H "Content-Type: application/json" \
  -d '{"use_finetuned": false}'
```

#### RAG Stats
```bash
curl http://localhost:8000/api/v1/rag/stats
```

### GitHub Webhook Integration

1. Create a GitHub App with PR read permissions
2. Set webhook URL to `https://your-domain.com/api/v1/review/github-webhook`
3. The system auto-reviews PRs on `opened`, `synchronize`, and `reopened` events

---

## API Documentation

Interactive API docs available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Response Format

```json
{
  "review_id": "a1b2c3d4",
  "pr_number": 1,
  "repository": "my-repo",
  "summary": "Found 6 issues (3 critical, 2 high, 1 low) in PR: Add authentication",
  "comments": [
    {
      "id": "a1b2c3d4-sec-001",
      "severity": "critical",
      "category": "security",
      "file_path": "auth.py",
      "line_number": 5,
      "message": "Hardcoded password detected. Use environment variables or a secrets manager.",
      "confidence_score": 0.95,
      "model_used": "ast-analyzer"
    }
  ],
  "overall_score": 38.0,
  "processing_time_ms": 1250,
  "model_version": "groq-llama3-8b"
}
```

---

## Screenshots

<p align="center">
  <img src="assets/screenshot-dashboard.png" alt="Dashboard" width="700">
</p>

<p align="center">
  <img src="assets/screenshot-results.png" alt="Review Results" width="700">
</p>

---

## Project Structure

```
code-review-assistant/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic settings
│   │   │   └── logging.py         # Structured logging
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic models
│   │   ├── routers/
│   │   │   └── review.py          # API endpoints
│   │   ├── services/
│   │   │   ├── code_analyzer.py   # AST + security scanner
│   │   │   ├── llm_service.py     # Groq API integration
│   │   │   ├── rag_engine.py      # FAISS vector search
│   │   │   └── review_orchestrator.py  # Pipeline coordinator
│   │   ├── main.py                # FastAPI app entry
│   │   └── __init__.py
│   ├── data/
│   │   └── vector_store/          # FAISS index files
│   ├── scripts/
│   │   └── test_api.py            # API test suite
│   ├── .env                       # Environment variables
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   └── index.html                 # Interactive dashboard
├── assets/
│   └── screenshots/               # Project screenshots
└── README.md
```

---

## Future Enhancements

- [ ] **Multi-language support** — JavaScript, TypeScript, Go, Rust analysis
- [ ] **Diff-aware reviews** — Only review changed lines in PRs
- [ ] **Review learning loop** — Feedback-based model improvement
- [ ] **Benchmark suite** — Compare against SonarQube, CodeClimate
- [ ] **CI/CD integration** — GitHub Actions, GitLab CI plugins
- [ ] **Team analytics** — Track code quality trends over time
- [ ] **Custom rule engine** — User-defined security/quality rules
- [ ] **Slack/Discord notifications** — Alert channels on critical issues

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ for the developer community
</p>
