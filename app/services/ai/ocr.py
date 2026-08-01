"""本地 RapidOCR 服务 — 混合「OCR 保真 → LLM 结构化」方案的文字抽取层。

简历 PDF/图片上传时，先用本地 RapidOCR 快速抽取文字（快、免费、无幻觉、字符
保真），OCR 输出过短（低于 ``settings.ocr_fallback_threshold``）或不可用时，
由调用方（``resume_parser.extract_text_from_file``）回退到 MIMO 多模态视觉，
保证准确率不降。

RapidOCR 未安装 / 加载失败时，本模块的 ``ocr_*`` 函数返回空串，调用方自然走
视觉，不影响原流程。模块懒加载单例，模式与 ``services/rag/reranker.py`` 一致。
"""

from __future__ import annotations

import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    """懒加载 RapidOCR 单例。返回 None/False 表示不可用（未安装或加载失败）。"""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
            logger.info("本地 RapidOCR 加载完成（混合 OCR+LLM 文字保真层）")
        except Exception as e:  # ImportError 或模型加载失败
            logger.warning("RapidOCR 不可用（%s），OCR 路径降级为多模态视觉", e)
            _engine = False
    return _engine


def _result_to_text(result) -> str:
    """把 RapidOCR 的 result 列表转成拼接文本。

    result 形如 [[box, text, score], ...]；防御不同版本结构差异。
    """
    if not result:
        return ""
    lines = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = item[1]
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def ocr_image_bytes(image_bytes: bytes) -> str:
    """对单张图片字节（PNG/JPEG 等）做 OCR，返回拼接文本。失败/不可用返回空串。"""
    engine = _get_engine()
    if not engine:
        return ""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as im:
            arr = np.array(im.convert("RGB"))
        result, _elapse = engine(arr)
        return _result_to_text(result)
    except Exception as e:
        logger.warning("RapidOCR 识别单张图片失败: %s", e)
        return ""


def ocr_pdf(
    file_path: str,
    max_pages: int = 6,
    zoom: float = 2.0,
    concurrency: int = 4,
) -> str:
    """对 PDF 前若干页做 OCR，返回拼接文本。失败/不可用返回空串。

    逐页并行识别（RapidOCR 推理是 CPU 密集，线程池并发），
    6 页从 ~20s 压到 ~5s。onnxruntime Session 支持并发 infer。
    """
    engine = _get_engine()
    if not engine:
        return ""
    try:
        from app.services.ai.vision import render_pdf_pages_to_png
        pages = render_pdf_pages_to_png(file_path, max_pages=max_pages, zoom=zoom)
        if not pages:
            return ""
        with ThreadPoolExecutor(max_workers=min(concurrency, len(pages))) as pool:
            parts = list(pool.map(ocr_image_bytes, pages))
        return "\n".join(p for p in parts if p.strip()).strip()
    except Exception as e:
        logger.warning("RapidOCR 识别 PDF 失败: %s", e)
        return ""
