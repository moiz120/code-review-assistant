"""
Application configuration using Pydantic Settings.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )
    
    # GitHub App Configuration
    github_app_id: str = ""
    github_private_key_path: str = ""
    github_webhook_secret: str = ""
    
    # Model Configuration
    model_path: str = "./models/codellama-finetuned"
    base_model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    use_unified_model: bool = True
    
    # Groq API Configuration
    groq_api_key: str = ""
    
    # RAG Configuration
    vector_store_path: str = "./data/vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_retrieval: int = 3
    
    # Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Hugging Face Token
    hf_token: str = ""
    
    @property
    def is_github_configured(self) -> bool:
        return bool(self.github_app_id and self.github_private_key_path)


settings = Settings()