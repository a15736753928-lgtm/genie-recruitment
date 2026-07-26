"""时间序列化工具 —— 统一给前端下发带 Z 后缀的 UTC 时间。

背景：本项目所有 ORM 时间列都是 ``Column(DateTime, default=datetime.utcnow)``，
即 **naive UTC**（不带 tzinfo）。直接 ``dt.isoformat()`` 得到的是
``2026-07-27T06:00:00``，浏览器 ``new Date()`` 会把它当作**本地时间**解析 ——
在 CST(UTC+8) 下刚创建的记录会显示成「8 小时前」。

因此所有对外序列化的时间字段一律走 ``iso_utc()``，而不是裸 ``.isoformat()``。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Union


def iso_utc(dt: Optional[Union[datetime, date]]) -> str:
    """把 naive UTC datetime 序列化为带 Z 后缀的 ISO 字符串。

    - ``None`` → 空字符串（前端按「无值」处理）
    - 已带时区信息（``Z`` / ``+00:00`` / ``+08:00``）的原样返回，不重复加 Z
    - ``date``（无时间部分）原样返回 ``YYYY-MM-DD``，加 Z 反而会让前端解析出错
    """
    if not dt:
        return ""
    # date 不是 datetime 的实例判断要放在前面：datetime 是 date 的子类
    if not isinstance(dt, datetime):
        return dt.isoformat()
    text = dt.isoformat()
    if text.endswith("Z") or text.endswith("+00:00"):
        return text
    # 已有形如 +08:00 / -05:00 的偏移量
    if len(text) >= 6 and text[-6] in "+-" and text[-3] == ":":
        return text
    return text + "Z"
