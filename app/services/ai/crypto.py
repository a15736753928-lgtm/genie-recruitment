"""API Key 加解密 —— Fernet 对称加密。

密钥来源：环境变量 SETTINGS_SECRET_KEY，未设置则自动生成（重启后旧密文失效）。
"""
from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache

from cryptography.fernet import Fernet


def _derive_fernet_key() -> bytes:
    raw = os.getenv("SETTINGS_SECRET_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.urandom(32).hex()
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


@lru_cache()
def _get_fernet() -> Fernet:
    return Fernet(_derive_fernet_key())


def encrypt_key(plain: str) -> str:
    """加密 API Key。"""
    if not plain:
        return ""
    return _get_fernet().encrypt(plain.encode("utf-8")).decode("utf-8")


def decrypt_key(cipher: str) -> str:
    """解密 API Key。密文为空或损坏时返回空字符串。"""
    if not cipher:
        return ""
    try:
        return _get_fernet().decrypt(cipher.encode("utf-8")).decode("utf-8")
    except Exception:
        return ""


def mask_key(plain: str) -> str:
    """脱敏显示：sk-****abcd"""
    if not plain:
        return ""
    if len(plain) <= 8:
        return plain[:4] + "****"
    return plain[:4] + "****" + plain[-4:]


SENTINEL_UNCHANGED = "__LLM_KEY_UNCHANGED__"
