from pathlib import Path

from pydantic_settings import BaseSettings


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    llm_provider: str = "mock"  # gemini | deepseek | qwen | mock

    llm_model_id: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""

    google_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    dashscope_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"
    qwen_model_sequence: str = (
        "qwen-plus,qwen-turbo,qwen-flash,qwen3.5-flash,"
        "qwen3.5-plus,qwen-max"
    )

    ieee_api_key: str = ""

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "scholar_papers"

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM settings (all free-tier Gemini — each model has separate 20 RPD quota)
    primary_model: str = "gemini-2.5-flash"
    fallback_model: str = "gemini-2.0-flash"
    tertiary_model: str = "gemini-2.0-flash-lite"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_request_timeout: float = 25.0
    llm_unavailable_cache_seconds: float = 300.0

    # Paper search
    max_papers: int = 10
    top_k_results: int = 10

    # Agent
    max_rewrite_retries: int = 2
    hallucination_threshold: float = 0.3
    max_hallucination_retries: int = 5

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    model_config = {
        "env_file": str(BACKEND_ROOT / ".env"),
        "env_file_encoding": "utf-8-sig",
        "extra": "ignore",
    }


settings = Settings()
