# -*- coding: utf-8 -*-
"""统一的 MIMO 多模态识别服务。

取代旧的本地方案：
- 简历文字抽取：不再用 PyMuPDF 读 PDF 文字层，而是把页面渲染成图，交 MIMO
  多模态直接提取（扫描件与文本 PDF 一视同仁，单一入口，降低复杂度）。
- 头像性别识别：不再用本地 ONNX 分类器 + cv2，直接调 MIMO 视觉识别。

所有调用走 ``llm_chat``（OpenAI 兼容 messages，图片用 data URL），与测试脚本
实测一致：mimo-v2.5 支持 image_url 多模态输入。配置见 ``settings.vision_*``。

对外提供的函数契约（被 resume_parser / ingest_service 引用）：
- async: ``extract_text_from_vision``、``infer_gender_from_vision``、
  ``infer_gender_from_image_bytes_async``、``extract_text_from_images_async``
- sync:  ``render_pdf_pages_to_png``、``extract_images_from_file``、
  ``extract_text_from_image_bytes_sync``、``extract_text_from_images_sync``
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Optional

import fitz  # PyMuPDF：仅用于渲染页面 / 提取内嵌图片（不做文字层抽取）
from PIL import Image

from openai import AsyncOpenAI

from app.config import get_settings
from app.prompts import render_prompt
from app.services.ai.router import llm_chat

logger = logging.getLogger(__name__)
settings = get_settings()

# 视觉客户端（懒加载单例）—— 独立于 Router，固定走 MIMO（xiaomimimo）端点。
# 文本解析已切到 DeepSeek，Router 客户端池不再指向 MIMO；视觉必须用独立客户端，
# 否则 model=mimo-v2.5 会打到 DeepSeek 端点而 404。
_vision_client: Optional[AsyncOpenAI] = None


def _get_vision_client() -> Optional[AsyncOpenAI]:
    global _vision_client
    if _vision_client is not None:
        return _vision_client
    if not settings.vision_api_key:
        return None
    _vision_client = AsyncOpenAI(
        api_key=settings.vision_api_key,
        base_url=settings.vision_base_url or "https://api.xiaomimimo.com/v1",
        timeout=60.0,
        max_retries=0,
    )
    return _vision_client

# 多页识别时最多渲染的页数 & 分辨率（清晰度与 token 成本的折中）
MAX_PAGES = 6
ZOOM = 2.0


# ── sync → async 桥 ──────────────────────────────────────

def _run_async(coro) -> object:
    """在同步上下文里运行一个 coroutine。

    供 ingest_service 等非 async 后台线程调用多模态识别用。若当前线程已有一个
    运行中的事件循环（罕见，通常只在已有 EventLoop 的线程里被调用），走
    run_coroutine_threadsafe，否则直接用 asyncio.run 新建循环。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # 已有运行中循环：把任务丢到该循环并阻塞等待结果
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# ── 图片编码 ─────────────────────────────────────────────

def _to_image_url(image_bytes: bytes, mime: str = "image/png") -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _to_content(text: str, images: list[dict]) -> list[dict]:
    return [{"type": "text", "text": text}] + images


def _bytes_to_png(data: bytes) -> Optional[bytes]:
    """把原始图片字节统一转成 PNG（保兼容多模态输入）。"""
    try:
        with Image.open(io.BytesIO(data)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None


def _image_file_to_png(file_path: str) -> Optional[bytes]:
    try:
        with open(file_path, "rb") as fh:
            return _bytes_to_png(fh.read())
    except Exception:
        return None


# ── PDF / 图片 → 图像字节 ────────────────────────────────

def render_pdf_pages_to_png(file_path: str, max_pages: int = MAX_PAGES, zoom: float = ZOOM) -> list[bytes]:
    """把 PDF 前若干页渲染成 PNG 列表（多模态识别的输入图）。"""
    doc = fitz.open(file_path)
    pngs: list[bytes] = []
    try:
        for page in doc[:max_pages]:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pngs.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pngs


def extract_images_from_file(file_path: str) -> list[bytes]:
    """提取文件内嵌的原始图片字节列表（PDF 前的若干页；直接图片文件则原样返回）。"""
    ext = file_path.lower() if file_path else ""
    if ext.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")):
        try:
            with open(file_path, "rb") as fh:
                return [fh.read()]
        except Exception:
            return []
    if not ext.endswith(".pdf"):
        return []
    images: list[bytes] = []
    try:
        doc = fitz.open(file_path)
        try:
            for page in doc[:2]:
                for img in page.get_images():
                    base = doc.extract_image(img[0])
                    data = base and base.get("image")
                    if data:
                        images.append(data)
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("提取 PDF 内嵌图片失败 %s: %s", file_path, exc)
    return images


# ── 简历文字提取 ─────────────────────────────────────────

async def extract_text_from_images_async(pages: list[bytes]) -> str:
    """对一组页面/图片字节做多模态文字识别，拼接返回。失败页跳过。"""
    parts: list[str] = []
    for idx, png in enumerate(pages):
        png_bytes = png if _is_png(png) else (_bytes_to_png(png) or b"")
        if not png_bytes:
            continue
        prompt = render_prompt("ai_services/vision.md", {
            "page_label": f"（第 {idx + 1} 页）" if len(pages) > 1 else "",
        })
        raw = await _llm_with_retry(
            _to_content(prompt, [_to_image_url(png_bytes)]),
            max_tokens=4000, what=f"多模态文字提取 (页 {idx + 1})",
        )
        if raw:
            parts.append(raw)
    return "\n".join(parts).strip()


async def _llm_with_retry(content: list[dict], *, max_tokens: int, what: str) -> str:
    """调用多模态 LLM 并处理其不稳定性。

    MIMO 是推理模型：可能 (a) reasoning_content 把 token 预算吃光导致 content 为空，
    (b) 偶发返回 400 role not supported。两者都通过「非内容本身问题」来判断并重试，
    空输出重试最多 3 次、网络/服务端错误基于 router 已有重试。避免把偶发空输出当成功。
    """
    full_messages = [{"role": "user", "content": content}]
    model = settings.vision_model
    client = _get_vision_client()
    for attempt in range(3):
        try:
            if client is not None:
                resp = await client.chat.completions.create(
                    model=model, messages=full_messages,
                    temperature=0.0, max_tokens=max_tokens,
                )
                raw = (resp.choices[0].message.content or "").strip()
            else:
                raw = await llm_chat(
                    full_messages,
                    model=model, temperature=0.0, max_tokens=max_tokens,
                )
        except Exception as exc:
            msg = str(exc)
            if "role is not supported" in msg:
                # MIMO 偶发 400，非结构问题——重试
                logger.warning("%s 收到 role 400 (attempt %d): %s", what, attempt + 1, exc)
                await asyncio.sleep(1.0)
                continue
            logger.warning("%s 调用失败 (attempt %d): %s", what, attempt + 1, exc)
            if attempt < 2:
                await asyncio.sleep(1.0)
                continue
            return ""
        if raw and raw.strip():
            return raw.strip()
        # 空输出：很可能是推理占满预算，重试加大稳定性
        logger.warning("%s 返回空内容，重试 (attempt %d)", what, attempt + 1)
        await asyncio.sleep(1.0)
    return ""


def _is_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


async def extract_text_from_vision(file_path: str) -> tuple[str, Optional[str]]:
    """用 MIMO 多模态从文件（PDF/图片）提取简历文字。返回 (text, error)。

    取代原 ``resume_parser.extract_text_from_file`` 的 PDF 文字层路径，
    是简历文本抽取的唯一入口。
    """
    if not file_path:
        return "", "简历文件不存在"
    if not settings.vision_enabled or not settings.vision_model:
        return "", "未配置多模态模型（vision）"

    ext = file_path.lower()
    try:
        if ext.endswith(".pdf"):
            pngs = render_pdf_pages_to_png(file_path)
        elif ext.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            png = _image_file_to_png(file_path)
            pngs = [png] if png else []
        elif ext.endswith((".docx", ".doc")):
            return "", "Word 文档请转换为 PDF 或图片后再上传（多模态识别需要图像输入）"
        else:
            return "", f"不支持的文件格式: {ext}"
    except Exception as exc:
        logger.exception("多模态提取时解析文件失败: %s", file_path)
        return "", f"文件处理失败: {exc}"

    if not pngs:
        return "", "未能渲染文件为图像，无法进行多模态识别"

    text = await extract_text_from_images_async(pngs)
    if not text:
        return "", "多模态未能提取到简历文字，请检查文件内容"
    return text, None


def extract_text_from_image_bytes_sync(image_bytes: bytes) -> str:
    """同步封装：对单张图片字节做多模态文字识别（供 ingest_service 等）。"""
    png = _bytes_to_png(image_bytes)
    if not png:
        return ""
    return _run_async(extract_text_from_images_async([png]))


def extract_text_from_images_sync(pages: list[bytes]) -> str:
    """同步封装：对页面/图片字节列表做多模态文字识别。"""
    if not pages:
        return ""
    return _run_async(extract_text_from_images_async(pages))


# ── 头像性别识别 ─────────────────────────────────────────

def _normalize_gender(answer: Optional[str]) -> Optional[str]:
    text = (answer or "").strip().replace(" ", "")
    if text in ("男", "女"):
        return text
    if "男" in text and "女" not in text:
        return "男"
    if "女" in text and "男" not in text:
        return "女"
    return None


async def infer_gender_from_image_bytes_async(image_bytes: bytes) -> Optional[str]:
    """识别头像图片字节的性别。返回 男/女/None。"""
    if not settings.vision_enabled or not settings.vision_model:
        return None
    png = _bytes_to_png(image_bytes)
    if not png:
        return None
    raw = await _llm_with_retry(
        _to_content(
            "这是一张中文简历中的证件照或头像。请根据照片中人物的典型外貌特征判断性别。" \
            "只回答一个字：男、女或未知。注意：直接给最终答案，不要解释。",
            [_to_image_url(png)],
        ),
        # 必须与文字提取一致用 4000：MIMO 是推理模型，reasoning_content 会先占
        # token，max_tokens=100 时思维链吃光预算 → content 为空 → 触发重试刷屏。
        # max_tokens 是上限不是实际量，性别只输出一个字，成本几乎不变。
        max_tokens=4000, what="多模态性别识别",
    )
    return _normalize_gender(raw)


async def infer_gender_from_vision(file_path: str) -> Optional[str]:
    """用 MIMO 视觉识别简历头像性别（直接文件路径入口）。"""
    images = extract_images_from_file(file_path or "")
    if not images:
        return None
    # 取最大一张作为头像
    best, best_area = None, -1
    for data in images:
        try:
            with Image.open(io.BytesIO(data)) as im:
                area = im.size[0] * im.size[1]
        except Exception:
            continue
        if area > best_area:
            best, best_area = data, area
    if not best:
        return None
    return await infer_gender_from_image_bytes_async(best)
