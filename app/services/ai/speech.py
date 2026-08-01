# -*- coding: utf-8 -*-
"""云端面试录音转写服务 —— 替代本地 FunASR。

走 MIMO `mimo-v2.5` 通用版全模态模型的 ``input_audio`` 音频输入，单次即可完整
转写长录音（实测 30 分钟整段无截断，自动标注「面试官/候选人」说话人），远超
专用 ASR 版（8192 token 上下文，30 分钟直接 400 拒绝）。

流程：
1. 收到任意格式音频/视频字节（wav/mp3/m4a/mp4…）；
2. ffmpeg 统一转码为 16kHz 单声道低码率 mp3（82MB 30 分钟 wav → ~5MB，base64
   约 6.8MB 云端才收）；视频自动抽音轨，无需特殊分支；
3. base64 → ``input_audio`` 调 ``mimo-v2.5`` 完整转写；
4. 空输出 / 400 / 超时重试 3 次（MIMO 是推理模型，思维链可能吃满 token）。

配置复用 ``settings.vision_*``（base_url 即 api.xiaomimimo.com/v1，model 默认
mimo-v2.5），不新增配置项。ffmpeg 为运行环境依赖（生产部署需安装）。

对外函数：``transcribe_audio_bytes``（async）、``transcribe_audio_bytes_sync``。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Optional

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# 音频客户端（懒加载单例）—— 与视觉客户端同一 MIMO 端点，独立实例便于单独调超时。
_audio_client: Optional[AsyncOpenAI] = None


def _get_audio_client() -> Optional[AsyncOpenAI]:
    global _audio_client
    if _audio_client is not None:
        return _audio_client
    if not settings.vision_api_key:
        return None
    _audio_client = AsyncOpenAI(
        api_key=settings.vision_api_key,
        base_url=settings.vision_base_url or "https://api.xiaomimimo.com/v1",
        timeout=600.0,  # 长录音转写可能跑几分钟
        max_retries=0,
    )
    return _audio_client


# ── ffmpeg 转码 ──────────────────────────────────────────

def transcode_to_mp3(audio_bytes: bytes, timeout: int = 300) -> bytes:
    """把任意音频/视频字节统一转成 16kHz 单声道低码率 mp3（同步，供 to_thread 调用）。

    ffmpeg 自动探测输入容器并抽取音频轨（视频直接抽音轨）。转码失败返回 b""。
    """
    if not audio_bytes:
        return b""
    fd_in, in_path = tempfile.mkstemp(suffix=".in")
    fd_out, out_path = tempfile.mkstemp(suffix=".mp3")
    try:
        with os.fdopen(fd_in, "wb") as f:
            f.write(audio_bytes)
        os.close(fd_out)  # 关闭后让 ffmpeg 覆盖写入
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path, "-ac", "1", "-ar", "16000", "-b:a", "24k", out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not os.path.exists(out_path):
            logger.error("ffmpeg 转码失败: %s", (r.stderr or "")[-800:])
            return b""
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── 云端转写 ─────────────────────────────────────────────

# 转写提示词：完整对话稿 + 说话人标注（对下游抽问答/评分最有价值）。
_TRANSCRIBE_PROMPT = (
    "这是一段面试录音。请把它完整转写成对话文字稿，标注说话人（面试官/候选人），"
    "不要总结、不要省略任何内容。"
)

# 30 分钟全文转写实测约 9500 字符（≈7k token），留足余量防截断。
MAX_COMPLETION_TOKENS = 12000


def _today_str() -> str:
    return datetime.now().strftime("%A, %B %d, %Y")


def _system_prompt() -> str:
    return (
        "You are MiMo, an AI assistant developed by Xiaomi. "
        f"Today is date: {_today_str()}. Your knowledge cutoff date is December 2024."
    )


async def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """云端转写音频字节为完整对话稿（async，interview.py 后台任务直接 await）。"""
    client = _get_audio_client()
    if client is None:
        logger.error("云端转写不可用：未配置 VISION_API_KEY")
        return "[转写失败：未配置云端语音服务]"

    mp3 = await asyncio.to_thread(transcode_to_mp3, audio_bytes)
    if not mp3:
        return "[转写完成，但未识别到语音内容]"
    b64 = base64.b64encode(mp3).decode("ascii")
    logger.info("云端转写输入 base64 %.1f MB", len(b64) / 1024 / 1024)

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": [{"type": "input_audio",
                                      "input_audio": {"data": b64}}]},
        {"role": "user", "content": _TRANSCRIBE_PROMPT},
    ]

    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=settings.vision_model,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("云端转写调用失败 (attempt %d): %s", attempt + 1, exc)
            await asyncio.sleep(1.0)
            continue
        if raw and raw.strip():
            return raw.strip()
        # 空输出：推理思维链可能吃满预算，重试
        logger.warning("云端转写返回空内容，重试 (attempt %d)", attempt + 1)
        await asyncio.sleep(1.0)
    return "[转写完成，但未识别到语音内容]"


def transcribe_audio_bytes_sync(audio_bytes: bytes) -> str:
    """同步封装（供非 async 后台线程调用）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(transcribe_audio_bytes(audio_bytes))
    future = asyncio.run_coroutine_threadsafe(transcribe_audio_bytes(audio_bytes), loop)
    return future.result()
