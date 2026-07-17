"""Determine a Chinese university's admission tier for rule-based scoring.

Tiers: 清北 / 985 / 211 / 一本 / 二本 / 民办本 / 专科 / 未知.

- 清北 / 985 / 211: decided by the official hardcoded lists (100% accurate,
  no network). Live web search was tried but Bing's static HTML doesn't expose
  result snippets reliably, and an LLM reading noisy snippets misclassified
  some schools (e.g. 郑州大学→985, 河南工业大学→211). Official lists fix that.
- 专科: decided from the school name (职业技术/高等专科… → 专科).
- 一本 / 二本 / 民办本 / 未知: decided by DeepSeek from its own knowledge of
  Chinese universities (tested 7/7 correct on borderline schools), which is
  more reliable than scraping search-engine result pages.

Results are cached per school name.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

TIERS = ("清北", "985", "211", "一本", "二本", "民办本", "专科", "未知")
_tier_cache: dict[str, str] = {}

# 985 工程高校（39 所，官方名单）
_985_SCHOOLS = (
    "清华大学", "北京大学", "中国人民大学", "北京航空航天大学", "北京理工大学",
    "中国农业大学", "北京师范大学", "中央民族大学", "南开大学", "天津大学",
    "大连理工大学", "东北大学", "吉林大学", "哈尔滨工业大学", "复旦大学",
    "同济大学", "上海交通大学", "华东师范大学", "南京大学", "东南大学",
    "浙江大学", "中国科学技术大学", "厦门大学", "山东大学", "武汉大学",
    "华中科技大学", "湖南大学", "中南大学", "中山大学", "华南理工大学",
    "四川大学", "重庆大学", "电子科技大学", "西安交通大学", "西北工业大学",
    "西北农林科技大学", "兰州大学", "国防科技大学", "中国海洋大学",
)

# 211 工程高校中非 985 的（官方名单）。命中即 211。
_211_ONLY_SCHOOLS = (
    "北京交通大学", "北京工业大学", "北京科技大学", "北京化工大学", "北京邮电大学",
    "北京林业大学", "北京中医药大学", "北京外国语大学", "中国传媒大学", "中央财经大学",
    "对外经济贸易大学", "中国政法大学", "华北电力大学", "中国矿业大学", "中国石油大学",
    "中国地质大学", "天津医科大学", "河北工业大学", "太原理工大学", "内蒙古大学",
    "辽宁大学", "大连海事大学", "延边大学", "东北师范大学", "哈尔滨工程大学",
    "东北农业大学", "东北林业大学", "上海财经大学", "上海大学", "上海外国语大学",
    "华东理工大学", "东华大学", "第二军医大学", "苏州大学", "南京航空航天大学",
    "南京理工大学", "河海大学", "江南大学", "南京农业大学", "中国药科大学",
    "南京师范大学", "安徽大学", "合肥工业大学", "福州大学", "南昌大学",
    "郑州大学", "武汉理工大学", "华中师范大学", "华中农业大学", "中南财经政法大学",
    "湖南师范大学", "暨南大学", "华南师范大学", "广西大学", "海南大学",
    "西南大学", "西南交通大学", "西南财经大学", "四川农业大学", "贵州大学",
    "云南大学", "西北大学", "西安电子科技大学", "长安大学", "陕西师范大学",
    "第四军医大学",
)


def _match_school(name: str, names: tuple[str, ...]) -> bool:
    """True if ``name`` is (or is a branch/department of) any known school.

    Uses prefix match to tolerate suffixes like 郑州大学软件学院 / 北京大学医学部,
    but skips the 独立学院 pattern ``<大学名><地名>学院`` (e.g. 北京理工大学珠海学院)
    so it isn't wrongly elevated to its parent university's tier.
    """
    for s in names:
        if name == s:
            return True
        if name.startswith(s):
            if name.endswith("学院") and s.endswith("大学") and name != s:
                continue  # 独立学院，不算本校
            return True
    return False


def _quick_name_tier(name: str) -> Optional[str]:
    """Tiers derivable from the school name alone (no network)."""
    if re.search(r"(职业技术|职业学院|高等专科|专科学校)", name):
        return "专科"
    if "清华" in name or name == "北大" or "北京大学" in name:
        return "清北"
    return None


async def classify_school_tier(school_name: str) -> str:
    """Return the admission tier of ``school_name`` (cached)."""
    if not school_name:
        return "未知"
    key = school_name.strip()
    if key in ("未知", "null", "None", ""):
        return "未知"
    if key in _tier_cache:
        return _tier_cache[key]

    quick = _quick_name_tier(key)
    if quick:
        _tier_cache[key] = quick
        return quick
    if _match_school(key, _985_SCHOOLS):
        _tier_cache[key] = "985"
        return "985"
    if _match_school(key, _211_ONLY_SCHOOLS):
        _tier_cache[key] = "211"
        return "211"

    # 一本/二本/民办本/专科/未知：用 DeepSeek 自身知识判断（无需联网搜索）。
    try:
        client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        prompt = (
            f"判断「{key}」这所中国高校的招生层次。只回答一个词，从以下选一个："
            f"一本、二本、民办本、专科、未知。\n"
            f"规则：先看办学性质——民办/独立学院/转设民办=民办本；公办本科按多数省份"
            f"本科录取批次分一本/二本（不确定时归二本）；高职专科=专科；信息不足=未知。"
        )
        resp = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        ans = (resp.choices[0].message.content or "").strip()
        for t in ("一本", "二本", "民办本", "专科", "未知"):
            if t in ans:
                _tier_cache[key] = t
                return t
    except Exception as exc:
        logger.warning("school tier classification failed for %s: %s", key, exc)

    _tier_cache[key] = "未知"
    return "未知"
