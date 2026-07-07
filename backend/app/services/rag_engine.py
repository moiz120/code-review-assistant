"""
Retrieval-Augmented Generation Engine for Code Review.
Uses vector similarity search to retrieve relevant historical reviews.
"""
import os
import json
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
from app.models.schemas import HistoricalReview, ReviewCategory, Severity
from app.core.config import settings
from app.core.logging import logger


class RAGEngine:
    """
    RAG Engine for retrieving relevant historical code review context.
    Uses FAISS for efficient similarity search with sentence embeddings.
    """
    
    def __init__(self):
        self.logger = logger.bind(service="rag_engine")
        self.embedding_model = None
        self.index = None
        self.reviews: List[HistoricalReview] = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
        self._load_model()
        self._load_or_create_index()
    
    def _load_model(self):
        """Load the sentence transformer embedding model."""
        self.logger.info("loading_embedding_model", model=settings.embedding_model)
        try:
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            self.logger.info("embedding_model_loaded")
        except Exception as e:
            self.logger.error("failed_to_load_embedding_model", error=str(e))
            raise
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        index_path = os.path.join(settings.vector_store_path, "faiss_index.bin")
        reviews_path = os.path.join(settings.vector_store_path, "reviews.json")
        
        os.makedirs(settings.vector_store_path, exist_ok=True)
        
        if os.path.exists(index_path) and os.path.exists(reviews_path):
            self.logger.info("loading_existing_index")
            self.index = faiss.read_index(index_path)
            with open(reviews_path, 'r') as f:
                data = json.load(f)
                self.reviews = [HistoricalReview(**r) for r in data]
            self.logger.info("index_loaded", review_count=len(self.reviews))
        else:
            self.logger.info("creating_new_index")
            self.index = faiss.IndexFlatIP(self.dimension)
            self._seed_with_sample_data()
    
    def _seed_with_sample_data(self):
        """Seed the vector store with sample high-quality reviews."""
        self.logger.info("seeding_with_sample_data")
        
        sample_reviews = [
            {
                "id": "sample-001",
                "repository": "general",
                "category": ReviewCategory.FUNCTIONAL,
                "code_snippet": "def process_data(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result",
                "review_comment": "Consider using list comprehension for better readability: [item * 2 for item in data]",
                "severity": Severity.LOW,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-002",
                "repository": "general",
                "category": ReviewCategory.SECURITY,
                "code_snippet": "import os\npassword = 'admin123'",
                "review_comment": "Hardcoded credentials detected. Use environment variables or a secrets manager instead.",
                "severity": Severity.CRITICAL,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-003",
                "repository": "general",
                "category": ReviewCategory.FUNCTIONAL,
                "code_snippet": "def get_user(id):\n    query = 'SELECT * FROM users WHERE id = ' + str(id)",
                "review_comment": "SQL injection vulnerability! Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (id,))",
                "severity": Severity.CRITICAL,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-004",
                "repository": "general",
                "category": ReviewCategory.REFACTORING,
                "code_snippet": "class UserManager:\n    def __init__(self):\n        self.users = []\n    def add_user(self, user):\n        self.users.append(user)\n    def remove_user(self, user):\n        self.users.remove(user)",
                "review_comment": "Consider using a set for O(1) lookup instead of list for O(n) operations if order does not matter.",
                "severity": Severity.MEDIUM,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-005",
                "repository": "general",
                "category": ReviewCategory.DOCUMENTATION,
                "code_snippet": "def calculate(x, y):\n    return x + y",
                "review_comment": "Function lacks docstring. Add documentation explaining parameters and return value.",
                "severity": Severity.LOW,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-006",
                "repository": "general",
                "category": ReviewCategory.PERFORMANCE,
                "code_snippet": "result = []\nfor i in range(1000000):\n    result.append(i * 2)",
                "review_comment": "Use list comprehension for better performance: result = [i * 2 for i in range(1000000)]",
                "severity": Severity.MEDIUM,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-007",
                "repository": "general",
                "category": ReviewCategory.STYLE,
                "code_snippet": "if x == True:",
                "review_comment": "Use 'if x:' instead of 'if x == True:' for Pythonic style.",
                "severity": Severity.LOW,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-008",
                "repository": "general",
                "category": ReviewCategory.FUNCTIONAL,
                "code_snippet": "def divide(a, b):\n    return a / b",
                "review_comment": "Missing error handling for division by zero. Add try-except or check if b == 0.",
                "severity": Severity.HIGH,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-009",
                "repository": "general",
                "category": ReviewCategory.SECURITY,
                "code_snippet": "import pickle\ndata = pickle.loads(user_input)",
                "review_comment": "Unsafe deserialization! pickle can execute arbitrary code. Use json instead.",
                "severity": Severity.CRITICAL,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "sample-010",
                "repository": "general",
                "category": ReviewCategory.REFACTORING,
                "code_snippet": "def process():\n    data = fetch_data()\n    cleaned = clean_data(data)\n    analyzed = analyze_data(cleaned)\n    return analyzed",
                "review_comment": "Consider using a pipeline pattern or method chaining for better readability and testability.",
                "severity": Severity.LOW,
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        for review_data in sample_reviews:
            review = HistoricalReview(**review_data)
            self.add_review(review)
        
        self.logger.info("sample_data_seeded", count=len(sample_reviews))
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype('float32')
    
    def add_review(self, review: HistoricalReview):
        """Add a historical review to the vector store."""
        embedding = self.embed_text(review.code_snippet)
        review.embedding = embedding.tolist()
        self.index.add(embedding.reshape(1, -1))
        self.reviews.append(review)
        self.logger.info("review_added", review_id=review.id, category=review.category)
    
    def retrieve_similar(self, code_snippet: str, top_k: int = None, 
                         category_filter: Optional[ReviewCategory] = None) -> List[Tuple[HistoricalReview, float]]:
        if top_k is None:
            top_k = settings.top_k_retrieval
        
        self.logger.info("retrieving_similar_reviews", 
                        code_length=len(code_snippet), 
                        top_k=top_k,
                        category_filter=category_filter)
        
        query_embedding = self.embed_text(code_snippet)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k * 2)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.reviews):
                continue
            review = self.reviews[idx]
            if category_filter and review.category != category_filter:
                continue
            results.append((review, float(score)))
            if len(results) >= top_k:
                break
        
        self.logger.info("retrieval_complete", results_found=len(results))
        return results
    
    def get_context_for_review(self, code_snippet: str, 
                                category_hint: Optional[ReviewCategory] = None) -> str:
        similar_reviews = self.retrieve_similar(code_snippet, category_filter=category_hint)
        if not similar_reviews:
            return ""
        
        context_parts = ["Here are some similar code patterns and their reviews:"]
        for i, (review, score) in enumerate(similar_reviews, 1):
            context_parts.append("\n--- Example " + str(i) + " (similarity: " + str(round(score, 3)) + ") ---")
            context_parts.append("Category: " + review.category.value)
            context_parts.append("Code:\n" + review.code_snippet)
            context_parts.append("Review: " + review.review_comment)
        
        return "\n".join(context_parts)
    
    def save_index(self):
        """Save the FAISS index and reviews to disk."""
        os.makedirs(settings.vector_store_path, exist_ok=True)
        index_path = os.path.join(settings.vector_store_path, "faiss_index.bin")
        reviews_path = os.path.join(settings.vector_store_path, "reviews.json")
        
        faiss.write_index(self.index, index_path)
        
        reviews_data = []
        for review in self.reviews:
            review_dict = review.model_dump()
            if isinstance(review_dict.get('created_at'), datetime):
                review_dict['created_at'] = review_dict['created_at'].isoformat()
            reviews_data.append(review_dict)
        
        with open(reviews_path, 'w') as f:
            json.dump(reviews_data, f, indent=2)
        
        self.logger.info("index_saved", review_count=len(self.reviews))
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        category_counts = {}
        for review in self.reviews:
            cat = review.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "total_reviews": len(self.reviews),
            "dimension": self.dimension,
            "embedding_model": settings.embedding_model,
            "category_distribution": category_counts
        }


# Singleton instance
rag_engine = RAGEngine()