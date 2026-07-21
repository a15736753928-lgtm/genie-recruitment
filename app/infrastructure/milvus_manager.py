"""
Milvus Lite manager — dual-vector embedded database.

Blueprint alignment: Section 5.2
- Dense vector: 1024-dim FLOAT_VECTOR, HNSW COSINE index
- Sparse vector: SPARSE_FLOAT_VECTOR, SPARSE_INVERTED_INDEX IP
- Both in a single collection with kb_id filter field
- Text stored in PostgreSQL chunks table, linked via milvus_pk

IMPORTANT: After upgrading from the old 512-dim single-vector schema,
existing collection must be dropped and documents re-ingested.
"""

import logging
import os
import re
import threading
import time
from queue import Empty, Queue
from typing import Optional

# ── 抑制 gRPC keepalive "too_many_pings" 日志噪音 ──────────
# pymilvus 3.x 内嵌 gRPC 客户端 keepalive 间隔太短（10s），
# Milvus Lite 服务端会主动 GOAWAY。reset_client() 已做了
# 容错处理，这里只抑制 C-core 的日志噪音。
# 必须在 import pymilvus 之前设置。
if not os.environ.get("GRPC_VERBOSITY"):
    os.environ["GRPC_VERBOSITY"] = "ERROR"

from pymilvus import MilvusClient, DataType
from app.config import get_settings

logger = logging.getLogger("genie.milvus")
settings = get_settings()


# ── Windows 兼容补丁：milvus_lite 用 os.rename 做「原子提交」写 manifest/schema，
# 但 os.rename 在 Windows 上当目标已存在时会抛 WinError 183，导致建索引/建集合
# 直接失败 —— 表现为集合永远建不出来、检索/入库不可用。os.replace 语义与
# os.rename 一致但总是覆盖（POSIX 上行为也不变），是跨平台的「原子替换」正解。
# 只替换受影响的 milvus_lite 子模块里对 os 的引用，不动全局 os，可逆、隔离。
def _patch_milvus_lite_atomic_rename() -> None:
    import types

    class _OsReplaceShim(types.ModuleType):
        """代理真实 os，仅把 rename 重定向到 replace（覆盖式原子替换）。"""

        def __init__(self, real_os):
            super().__init__(real_os.__name__)
            self.__dict__["_real_os"] = real_os

        def __getattr__(self, name):
            real = self.__dict__["_real_os"]
            if name == "rename":
                return real.replace
            return getattr(real, name)

    shim = _OsReplaceShim(os)
    for mod_name in (
        "milvus_lite.storage.manifest",
        "milvus_lite.schema.persistence",
        "milvus_lite.db",
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            mod.os = shim  # type: ignore[attr-defined]
        except Exception as e:  # 子模块路径变动等，非致命
            logger.debug("milvus_lite rename 兼容补丁跳过 %s: %s", mod_name, e)


_patch_milvus_lite_atomic_rename()

_KB_ID_PATTERN = re.compile(r'^kb_[a-f0-9]{8}$')

# 可重入锁：ensure_collection() 持锁后会再调用 _get_client()，而 _get_client()
# 自身也要拿同一把锁。普通 Lock 不可重入 —— 同一线程二次 acquire 会永久自死锁，
# 锁再也不释放，后续所有 Milvus 调用（入库/检索/统计）随之全部卡死，
# 表现为文档一直「索引中」、分片恒为 0。必须用 RLock。
_lock = threading.RLock()
_client: Optional[MilvusClient] = None
_collection_ready: bool = False

# 写操作互斥锁：多线程并发 insert/delete 会通过同一个 gRPC channel
# 密集发送请求，压垮 Milvus Lite 内嵌服务端。串行化写入确保同一时刻
# 只有一个线程在做 Milvus 变更操作，配合 gRPC keepalive 参数彻底消除
# GOAWAY 风险。读操作（search）不加锁，允许并发。
_write_lock = threading.Lock()

# 单次 insert 最长等待；超时即判定连接已死，重置客户端后重试一次
_INSERT_TIMEOUT = 60.0


# ── gRPC keepalive 参数：pymilvus 默认 10s 发一次 ping，
# Milvus Lite 服务端会因 "too_many_pings" 主动 GOAWAY 踢断连接。
# 把 keepalive 间隔拉长到 60s，配合不限次数的无数据 ping，
# 既保持连接活性又不超过服务端容忍上限。
_MILVUS_GRPC_OPTIONS = {
    "grpc.keepalive_time_ms": 60000,          # ping 间隔 60s（默认 10s 太短）
    "grpc.keepalive_timeout_ms": 10000,       # ping ack 等待 10s
    "grpc.keepalive_permit_without_calls": True,  # 空闲时也发 keepalive
    "grpc.http2.max_pings_without_data": 0,   # 无限制（避免 ENHANCE_YOUR_CALM）
    "grpc.http2.min_time_between_pings_ms": 30000,  # 两次 ping 至少间隔 30s
}


def _get_client() -> MilvusClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = MilvusClient(
                    settings.milvus_db_path,
                    grpc_options=_MILVUS_GRPC_OPTIONS,
                )
    return _client


def reset_client() -> None:
    """丢弃当前客户端与就绪标志，下次调用会重建连接。

    Milvus Lite 内嵌 gRPC 服务端会因 keepalive「too_many_pings」主动 GOAWAY
    掉连接，缓存客户端此后所有调用都会永久挂起。出现超时/异常时调用本函数。
    """
    global _client, _collection_ready
    with _lock:
        old = _client
        _client = None
        _collection_ready = False
    # 异步关闭旧客户端，避免 close() 自身挂住阻塞调用方
    if old is not None:
        threading.Thread(target=_safe_close, args=(old,), daemon=True).start()


def _safe_close(client: MilvusClient) -> None:
    try:
        client.close()
    except Exception as e:
        logger.debug("关闭旧 Milvus 客户端失败（可忽略）: %s", e)


def ensure_collection() -> bool:
    """Create dual-vector collection if not exists. Returns True if ready."""
    global _collection_ready
    if _collection_ready:
        return True

    with _lock:
        if _collection_ready:
            return True
        try:
            client = _get_client()
            coll_name = settings.milvus_collection_name

            if client.has_collection(coll_name):
                _collection_ready = True
                return True

            # Create schema with dual vector fields
            schema = client.create_schema(
                auto_id=True,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            )
            schema.add_field(
                field_name="kb_id",
                datatype=DataType.VARCHAR,
                max_length=128,
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=settings.embedding_dim,  # 1024
            )
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )

            # pymilvus 3.x：索引通过 IndexParams 描述，且随 create_collection
            # 一起建。绝不能像旧版那样先 create_collection 再 create_index —
            # 一旦 create_index 失败，会残留一个「有集合、无索引」的半成品：
            # 重试时 has_collection 命中直接返回就绪，可写入却搜不出来。
            index_params = client.prepare_index_params()
            # Dense HNSW index (COSINE)
            index_params.add_index(
                field_name="dense_vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 256},
            )
            # Sparse inverted index (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            # Create collection + indexes atomically
            client.create_collection(
                collection_name=coll_name,
                schema=schema,
                index_params=index_params,
            )

            _collection_ready = True
            return True

        except Exception as e:
            # 失败时清掉可能残留的半成品集合，避免下次 has_collection 命中一个无索引的坏集合
            try:
                if client.has_collection(coll_name):
                    client.drop_collection(coll_name)
            except Exception:
                pass
            logger.error("Milvus 集合初始化失败: %s", e, exc_info=True)
            return False


def insert_vectors(
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[int, float]],
    kb_ids: list[str],
) -> list[int]:
    """Insert dual vectors (dense + sparse) into Milvus.

    Args:
        dense_vectors: List of 1024-dim float lists
        sparse_vectors: List of {token_id: weight} dicts
        kb_ids: kb_id per vector

    Returns:
        List of auto-generated integer IDs (milvus_pk). 失败返回 []。
    """
    if not dense_vectors:
        return []

    # Sanitize empty sparse dicts — Milvus Lite can hang/fail on {}
    safe_sparse = []
    for s_vec in sparse_vectors:
        if isinstance(s_vec, dict) and s_vec:
            safe_sparse.append(s_vec)
        else:
            safe_sparse.append({0: 0.0})

    coll_name = settings.milvus_collection_name
    data = []
    for d_vec, s_vec, kb_id in zip(dense_vectors, safe_sparse, kb_ids):
        data.append({
            "dense_vector": d_vec,
            "sparse_vector": s_vec,
            "kb_id": kb_id,
        })

    # 串行化写入：防止多线程并发 insert 压垮 Milvus Lite gRPC 服务端
    with _write_lock:
        # 最多尝试 2 次：第一次超时/异常 → 重置客户端 → 第二次用全新连接
        for attempt in range(2):
            if not ensure_collection():
                reset_client()
                continue
            ids = _insert_with_timeout(coll_name, data, _INSERT_TIMEOUT)
            if ids is not None:
                return ids
            # 超时或异常：连接已死，重置后重试
            logger.warning("Milvus insert 超时/失败，重置客户端后重试 (attempt=%d)", attempt + 1)
            reset_client()

        logger.error("Milvus insert 两次均失败，放弃")
        return []


def _insert_with_timeout(coll_name: str, data: list[dict], timeout: float) -> Optional[list[int]]:
    """在子线程内执行 insert，主线程最多等 timeout 秒。

    返回:
        list[int]  成功
        None      超时或异常（调用方应重置客户端）
    """
    result_q: Queue = Queue()

    def _worker():
        try:
            # 只在拿客户端指针时短暂持锁（_get_client 内部自管懒加载锁），
            # 绝不在持锁状态下执行会阻塞的 insert RPC——否则连接假死时
            # 本线程会永久持锁挂住，reset_client() 再也拿不到锁，形成死锁，
            # 所有入库线程随之全部卡在 indexing/85。
            client = _get_client()
            result = client.insert(collection_name=coll_name, data=data)
            ids = result.get("ids", [])
            result_q.put(ids if isinstance(ids, list) else list(ids))
        except Exception as e:
            logger.warning("Milvus insert 异常: %s", e)
            result_q.put(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        return result_q.get(timeout=timeout)
    except Empty:
        logger.warning("Milvus insert 超时（%ss），判定连接已死", timeout)
        return None


def _call_with_timeout(fn, timeout: float = 30.0, *args, **kwargs):
    """在子线程内执行一个 Milvus 调用，超时则返回 None 并触发客户端重置。"""
    result_q: Queue = Queue()

    def _worker():
        try:
            # 同 _insert_with_timeout：拿到客户端后立刻脱离锁再发 RPC，
            # 避免连接假死时持锁挂起，拖垮 reset_client() 与其余调用。
            client = _get_client()
            result_q.put(("ok", fn(client, *args, **kwargs)))
        except Exception as e:
            logger.warning("Milvus 调用异常: %s", e)
            result_q.put(("err", None))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        tag, val = result_q.get(timeout=timeout)
        return val if tag == "ok" else None
    except Empty:
        logger.warning("Milvus 调用超时（%ss），判定连接已死", timeout)
        reset_client()
        return None


def search_dense(
    query_vector: list[float],
    top_k: int = 10,
    kb_id: Optional[str] = None,
) -> list[dict]:
    """Dense vector search (COSINE similarity).

    Returns:
        [{id: milvus_pk, distance: float, kb_id: str}, ...]
    """
    if not ensure_collection():
        return []

    coll_name = settings.milvus_collection_name
    filter_expr = None
    if kb_id:
        if not _KB_ID_PATTERN.match(kb_id):
            raise ValueError(f"Invalid kb_id format: {kb_id[:20]}...")
        filter_expr = f'kb_id == "{kb_id}"'

    def _do(client):
        return client.search(
            collection_name=coll_name,
            data=[query_vector],
            anns_field="dense_vector",
            limit=top_k,
            filter=filter_expr,
            output_fields=["kb_id"],
            search_params={"ef": settings.search_ef},
        )

    results = _call_with_timeout(_do, 30.0)
    if results is None or not results or not results[0]:
        return []

    hits = []
    for hit in results[0]:
        hits.append({
            "id": hit["id"],
            "distance": hit["distance"],
            "kb_id": hit.get("entity", {}).get("kb_id", ""),
        })
    return hits


def search_sparse(
    sparse_vector: dict[int, float],
    top_k: int = 10,
    kb_id: Optional[str] = None,
) -> list[dict]:
    """Sparse vector search (IP — inner product).

    Args:
        sparse_vector: {token_id: weight} dict
        top_k: Number of results
        kb_id: Optional KB filter

    Returns:
        [{id: milvus_pk, distance: float, kb_id: str}, ...]
    """
    if not ensure_collection():
        return []

    coll_name = settings.milvus_collection_name
    filter_expr = None
    if kb_id:
        if not _KB_ID_PATTERN.match(kb_id):
            raise ValueError(f"Invalid kb_id format: {kb_id[:20]}...")
        filter_expr = f'kb_id == "{kb_id}"'

    def _do(client):
        return client.search(
            collection_name=coll_name,
            data=[sparse_vector],
            anns_field="sparse_vector",
            limit=top_k,
            filter=filter_expr,
            output_fields=["kb_id"],
        )

    results = _call_with_timeout(_do, 30.0)
    if results is None or not results or not results[0]:
        return []

    hits = []
    for hit in results[0]:
        hits.append({
            "id": hit["id"],
            "distance": hit["distance"],
            "kb_id": hit.get("entity", {}).get("kb_id", ""),
        })
    return hits


def delete_by_ids(ids: list[int]) -> bool:
    """Delete vectors by Milvus primary keys."""
    if not ids or not ensure_collection():
        return False

    coll_name = settings.milvus_collection_name

    with _write_lock:
        def _do(client):
            id_list = ", ".join(str(i) for i in ids)
            client.delete(collection_name=coll_name, filter=f"id in [{id_list}]")
            return True

        return bool(_call_with_timeout(_do, 30.0) or False)


def get_collection_stats() -> dict:
    """Get collection statistics."""
    if not ensure_collection():
        return {"row_count": 0}

    coll_name = settings.milvus_collection_name

    def _do(client):
        return client.get_collection_stats(coll_name)

    return _call_with_timeout(_do, 30.0) or {"row_count": 0}


def drop_collection() -> bool:
    """Drop and recreate the collection (for schema migration)."""
    global _collection_ready
    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        if client.has_collection(coll_name):
            client.drop_collection(coll_name)
        _collection_ready = False
        return ensure_collection()
    except Exception as e:
        print(f"[Milvus] Drop error: {e}")
        return False
