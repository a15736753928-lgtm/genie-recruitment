from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 旧模型名 / Pro 统一映射到 Flash（用户要求不用 Pro）
_LLM_MODEL_ALIASES: dict[str, str] = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-flash",
}


def normalize_llm_model(model: str | None) -> str:
    """Normalize legacy or Pro model ids to deepseek-v4-flash."""
    name = (model or "").strip()
    if not name:
        return "deepseek-v4-flash"
    return _LLM_MODEL_ALIASES.get(name.lower(), name)


class Settings(BaseSettings):
    # Database — set via DATABASE_URL / DATABASE_URL_SYNC in .env
    database_url: str = ""
    database_url_sync: str = ""

    # DeepSeek — set via DEEPSEEK_API_KEY in .env
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # Vision (resume portrait gender inference)
    vision_enabled: bool = True
    vision_model: str = "deepseek-v4-flash"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    # Upload
    upload_dir: str = os.path.join(_PROJECT_ROOT, "uploads")
    max_upload_size: int = 52428800  # 50MB

    # MinIO — object storage for uploaded documents (PDF/Word/MD/images)
    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "genie-recruitment"
    minio_secure: bool = False
    # When true, uploads go to MinIO. When false (e.g. MinIO unreachable), fall
    # back to the local upload_dir so the app stays usable in dev.
    minio_enabled: bool = True

    # Milvus
    milvus_db_path: str = "milvus_lite.db"
    milvus_collection_name: str = "hr_knowledge_chunks"
    embedding_dim: int = 1024  # BGE-M3 dense output dimension
    embedding_model: str = "BAAI/bge-m3"  # tokenizer name for BGE-M3
    embedding_device: str = "cuda"  # "cpu" or "cuda"

    # BGE-M3 ONNX
    bge_onnx_model_name: str = "gpahal/bge-m3-onnx-int8"
    bge_onnx_filename: str = "model_quantized.onnx"
    sparse_vector_enabled: bool = True

    # RAG — Chunking
    chunk_size: int = 500       # default chunk size (characters)
    chunk_overlap: int = 100    # overlap between chunks
    min_chunk_size: int = 40    # merge chunks shorter than this

    # RAG — Retrieval
    default_top_k: int = 10
    max_search_recall: int = 5000
    min_similarity: float = 0.0
    search_ef: int = 64               # HNSW search parameter
    rerank_enabled: bool = True
    rerank_top_k: int = 20   # candidates to fetch before reranking
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # CrossEncoder reranker

    # RAG — RRF (Reciprocal Rank Fusion)
    rrf_k: int = 60                   # RRF smoothing constant
    rrf_dense_weight: float = 0.7     # dense semantic match weight
    rrf_sparse_weight: float = 0.3    # sparse keyword match weight
    rrf_graph_weight: float = 0.25    # graph structure weight

    # RAG — Kuzu Graph Database
    # 图谱检索为三路混合召回的第三路（dense+sparse+graph）。默认关闭：
    # 检索由 dense+sparse+rerank 承担，图谱那路对结果无必需贡献，且新版 Kuzu
    # 路径/API 兼容问题会在启动刷错。需要 GraphRAG 时把下面三个开关置 True 即可，
    # 代码路径仍完整保留。
    kuzu_data_dir: str = "storage/kuzu_data"
    kuzu_enabled: bool = False
    graph_index_enabled: bool = False  # background graph indexing after ingestion

    # RAG — Community Detection
    community_enabled: bool = False

    # RAG — Ingestion
    ingest_batch_size: int = 64     # embedding batch size
    max_file_size: int = 20 * 1024 * 1024  # 20 MB

    # RAG — OCR (RapidOCR for image/PDF fallback)
    ocr_enabled: bool = True
    ocr_fallback_threshold: int = 100  # chars below which OCR is triggered

    # Agent OS — intelligent agent layer
    agent_os_enabled: bool = True
    memory_dir: str = "memory/"
    skills_dir: str = "skills/"
    max_context_tokens: int = 8000
    enable_reflection: bool = True
    enable_verification: bool = True
    enable_parallel_subagents: bool = False   # Phase 4
    max_subagents: int = 10                    # Phase 4
    plan_confirm_timeout: int = 300            # Phase 3 (5 minutes)

    @field_validator("deepseek_model", "vision_model", mode="before")
    @classmethod
    def _normalize_model_fields(cls, v):
        return normalize_llm_model(v if v else "deepseek-v4-flash")

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
