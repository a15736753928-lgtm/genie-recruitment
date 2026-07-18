"""Extract resume portrait photos and infer gender via vision or local face analysis."""

from __future__ import annotations

import base64
import io
import logging
import os
import urllib.request
import zipfile
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "face_models")
# Gender classifier (PaddlePaddle LCNet, exported to ONNX, ~2.4 MB).
# Fetched from the HuggingFace mirror (hf-mirror.com) which is reachable in CN.
GENDER_ONNX_FILE = "gender_lcnet.onnx"
GENDER_ONNX_URL = "https://hf-mirror.com/kunkunlin1221/face-gender-lcnet-050/resolve/main/gender_detection_lcnet_050.onnx"
# ONNX input is 112x112 RGB; label order verified empirically: 0=male, 1=female.
GENDER_INPUT_SIZE = 112


def _image_metrics(data: bytes) -> Optional[Tuple[int, int, int, float]]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
        if w <= 0 or h <= 0:
            return None
        return w * h, w, h, w / h
    except Exception:
        return None


def _is_likely_portrait(area: int, width: int, height: int, aspect: float) -> bool:
    if width < 80 or height < 80 or area < 8_000:
        return False
    if area > 300_000 or max(width, height) > 650:
        return False
    if aspect > 2.2 or aspect < 0.45:
        return False
    return True


def _pick_portrait(candidates: List[bytes]) -> Optional[bytes]:
    scored: List[Tuple[float, bytes]] = []
    for data in candidates:
        metrics = _image_metrics(data)
        if not metrics:
            continue
        area, width, height, aspect = metrics
        if not _is_likely_portrait(area, width, height, aspect):
            continue
        score = float(area)
        if 0.65 <= aspect <= 1.35:
            score *= 1.8
        if max(width, height) <= 320:
            score *= 1.4
        scored.append((score, data))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _extract_images_from_pdf(file_path: str) -> List[bytes]:
    from PyPDF2 import PdfReader

    images: List[bytes] = []
    reader = PdfReader(file_path)
    for page in reader.pages[:2]:
        for image in getattr(page, "images", []):
            if getattr(image, "data", None):
                images.append(image.data)
    return images


def _extract_images_from_docx(file_path: str) -> List[bytes]:
    images: List[bytes] = []
    with zipfile.ZipFile(file_path) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if lower.startswith("word/media/") and lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                images.append(archive.read(name))
    return images


def extract_portrait_from_file(file_path: str) -> Optional[bytes]:
    """Extract the most likely headshot image from a resume file."""
    if not file_path or not os.path.exists(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()
    images: List[bytes] = []

    try:
        if ext == ".pdf":
            images = _extract_images_from_pdf(file_path)
        elif ext == ".docx":
            images = _extract_images_from_docx(file_path)
        elif ext in (".jpg", ".jpeg", ".png", ".webp"):
            with open(file_path, "rb") as handle:
                return handle.read()
        else:
            return None
    except Exception as exc:
        logger.warning("Failed to extract portrait from %s: %s", file_path, exc)
        return None

    return _pick_portrait(images)


def _download_if_missing(filename: str, url: str) -> str:
    target = os.path.join(MODEL_DIR, filename)
    if not (os.path.exists(target) and os.path.getsize(target) > 0):
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info("Downloading model %s <- %s", filename, url)
        urllib.request.urlretrieve(url, target)
    return target


def _ensure_gender_model() -> str:
    return _download_if_missing(GENDER_ONNX_FILE, GENDER_ONNX_URL)


def _parse_gender_answer(answer: str) -> Optional[str]:
    text = (answer or "").strip().replace(" ", "")
    if text in ("男", "女"):
        return text
    if "未知" in text:
        return None
    if "男" in text and "女" not in text:
        return "男"
    if "女" in text:
        return "女"
    return None


def _guess_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return "image/webp"
    return "image/jpeg"


@lru_cache(maxsize=1)
def _load_gender_classifier():
    import onnxruntime as ort

    path = _ensure_gender_model()
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def infer_gender_local(image_bytes: bytes) -> Optional[str]:
    """Infer gender from a resume portrait via the ONNX LCNet classifier.

    Resume headshots (证件照) are tight crops where the face fills most of the
    frame, so the classifier is fed the whole picked portrait resized to 112x112
    — no separate face-detection model is needed (OpenCV 5.0 also dropped the
    Caffe reader we'd otherwise use).
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python-headless is not installed; local portrait gender inference disabled")
        return None

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None

    try:
        gender_sess = _load_gender_classifier()
    except Exception as exc:
        logger.warning("Failed to load gender ONNX model: %s", exc)
        return None

    # BGR -> RGB, resize to 112x112, /255, NCHW float32.
    face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, (GENDER_INPUT_SIZE, GENDER_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    x = arr.transpose(2, 0, 1)[None]  # 1x3x112x112

    in_name = gender_sess.get_inputs()[0].name
    logits = gender_sess.run(None, {in_name: x})[0][0]
    gender_index = int(np.argmax(logits))
    # Label order verified empirically: 0=male, 1=female.
    gender = "男" if gender_index == 0 else "女"
    logger.info("Local portrait gender inferred as %s (logits=%s)", gender, logits.tolist())
    return gender


def infer_gender_from_vision(image_bytes: bytes, client: Optional[OpenAI] = None) -> Optional[str]:
    """Try a vision-capable LLM when configured."""
    if not settings.vision_enabled or not settings.vision_model:
        return None

    llm_client = client or OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=60.0,
        max_retries=0,
    )
    mime = _guess_mime(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")

    try:
        response = llm_client.chat.completions.create(
            model=settings.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "这是一张中文简历中的证件照或头像。"
                            "请根据照片中人物的典型外貌特征判断性别。"
                            "只回答一个字：男、女或未知。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            }],
            temperature=0.1,
            max_tokens=16,
        )
        return _parse_gender_answer(response.choices[0].message.content or "")
    except Exception as exc:
        logger.warning("Vision gender inference failed: %s", exc)
        return None


def _vision_configured() -> bool:
    """True only when a real vision model is configured (not the bogus default)."""
    if not settings.vision_enabled or not settings.vision_model:
        return False
    if settings.vision_model == "deepseek-v4-flash":
        return False  # DeepSeek has no vision model; skip the failing call.
    return True


def infer_gender_from_portrait(image_bytes: bytes, client: Optional[OpenAI] = None) -> Optional[str]:
    # Local ONNX is the primary path: fully offline, fast, no API.
    gender = infer_gender_local(image_bytes)
    if gender in ("男", "女"):
        return gender
    # Fall back to a vision LLM only when one is genuinely configured.
    if _vision_configured():
        gender = infer_gender_from_vision(image_bytes, client=client)
        if gender in ("男", "女"):
            return gender
    return None


def infer_gender_from_resume_file(file_path: str, client: Optional[OpenAI] = None) -> Optional[str]:
    portrait = extract_portrait_from_file(file_path)
    if not portrait:
        return None
    return infer_gender_from_portrait(portrait, client=client)
