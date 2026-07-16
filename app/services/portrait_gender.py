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
MODEL_FILES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
}
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


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


def _ensure_face_models() -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, url in MODEL_FILES.items():
        target = os.path.join(MODEL_DIR, filename)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue
        logger.info("Downloading face model %s", filename)
        urllib.request.urlretrieve(url, target)
    return MODEL_DIR


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
def _load_face_nets():
    import cv2

    model_dir = _ensure_face_models()
    face_net = cv2.dnn.readNetFromCaffe(
        os.path.join(model_dir, "deploy.prototxt"),
        os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel"),
    )
    gender_net = cv2.dnn.readNetFromCaffe(
        os.path.join(model_dir, "gender_deploy.prototxt"),
        os.path.join(model_dir, "gender_net.caffemodel"),
    )
    return face_net, gender_net


def infer_gender_local(image_bytes: bytes) -> Optional[str]:
    """Infer gender from portrait using OpenCV face + gender models."""
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python-headless is not installed; local portrait gender inference disabled")
        return None

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None

    try:
        face_net, gender_net = _load_face_nets()
    except Exception as exc:
        logger.warning("Failed to load face models: %s", exc)
        return None

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf = 0.0
    best_box = None
    for index in range(detections.shape[2]):
        confidence = float(detections[0, 0, index, 2])
        if confidence < 0.45 or confidence <= best_conf:
            continue
        box = detections[0, 0, index, 3:7] * np.array([width, height, width, height])
        best_conf = confidence
        best_box = box.astype(int)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    face = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if face.size == 0:
        return None

    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    prediction = gender_net.forward()[0]
    gender_index = int(np.argmax(prediction))
    gender = "男" if gender_index == 0 else "女"
    logger.info("Local portrait gender inferred as %s (confidence=%.2f)", gender, best_conf)
    return gender


def infer_gender_from_vision(image_bytes: bytes, client: Optional[OpenAI] = None) -> Optional[str]:
    """Try a vision-capable LLM when configured."""
    if not settings.vision_enabled or not settings.vision_model:
        return None

    llm_client = client or OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
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


def infer_gender_from_portrait(image_bytes: bytes, client: Optional[OpenAI] = None) -> Optional[str]:
    gender = infer_gender_from_vision(image_bytes, client=client)
    if gender in ("男", "女"):
        return gender
    return infer_gender_local(image_bytes)


def infer_gender_from_resume_file(file_path: str, client: Optional[OpenAI] = None) -> Optional[str]:
    portrait = extract_portrait_from_file(file_path)
    if not portrait:
        return None
    return infer_gender_from_portrait(portrait, client=client)
