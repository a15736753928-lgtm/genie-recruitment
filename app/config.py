from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 旧 deepseek 模型名已弃用（2026-07-24）：deepseek-chat/deepseek-reasoner 现对应
# deepseek-v4-flash 的非思考/思考模式。文本解析统一用 deepseek-v4-flash（关闭思考）。
# 不再把 deepseek-* 映射到 mimo —— 文本走 DeepSeek，视觉走 MIMO，各用各的模型。
_LLM_MODEL_ALIASES: dict[str, str] = {}


def normalize_llm_model(model: str | None) -> str:
    """Normalize legacy model ids to mimo-v2.5."""
    name = (model or "").strip()
    if not name:
        return "mimo-v2.5"
    return _LLM_MODEL_ALIASES.get(name.lower(), name)


class Settings(BaseSettings):
    # Database — set via DATABASE_URL / DATABASE_URL_SYNC in .env
    database_url: str = ""
    database_url_sync: str = ""

    # LLM 文本解析 — DeepSeek（OpenAI 兼容协议）
    # set via DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL in .env
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # Vision 视觉理解 — MIMO 多模态（与文本 LLM 分开配置）
    # set via VISION_API_KEY / VISION_BASE_URL / VISION_MODEL in .env
    vision_enabled: bool = True
    vision_api_key: str = ""
    vision_base_url: str = "https://api.xiaomimimo.com/v1"
    vision_model: str = "mimo-v2.5"

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

    # RAG — Ingestion
    ingest_batch_size: int = 64     # embedding batch size
    max_file_size: int = 20 * 1024 * 1024  # 20 MB

    # 简历 PDF/图片抽取 — 本地 OCR（RapidOCR）优先，不足则回退多模态视觉
    ocr_enabled: bool = True
    ocr_fallback_threshold: int = 100  # OCR 输出低于此字符数视为不可靠，回退视觉

    @field_validator("deepseek_model", "vision_model", mode="before")
    @classmethod
    def _normalize_model_fields(cls, v):
        return normalize_llm_model(v if v else "mimo-v2.5")

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
