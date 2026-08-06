# -*- coding: utf-8 -*-
"""Offer 正式 PDF 生成：在原始模板 PDF 的填空位置写入字段值（PyMuPDF 直接填充）。

模板来源：Offer 通用模板（3 页录用通知函），放置于 templates/offer_template.pdf。
填充策略：
  - after 模式：page.search_for(锚文本) 定位锚文本矩形，值写在锚文本右侧；
  - after_left 模式：值写在锚文本左侧（年终奖「____ 个月」）；
  - bracket 模式：锚文本是〔〕占位符（如〔公司名称〕），先涂白占位符再写入值。
模板为固定文件，字段坐标由本模块标定（锚文本动态定位 + 相对偏移 dx/dy 微调）。
若替换模板文件，需重新标定坐标。

无数据源的字段留空（不写入，保持原下划线）。
"""
import io
import logging
from datetime import date
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "offer_template.pdf"

# 填写值颜色：普通字段深蓝（电子填写惯例），编号/日期等函件要素黑色
_FILL_COLOR = (0.02, 0.25, 0.65)
_ISSUE_COLOR = (0.05, 0.05, 0.05)

# ── 中文字体：优先用微软雅黑（与模板字体一致）──
# msyh.ttc 是 TTF 集合，PyMuPDF 直接嵌入时 pdfjs 渲染成方块；用 fontTools 提取单个 TTF
# 后再嵌入可正常渲染且字体与模板完全一致。fontTools 缺失或提取失败回退内置 china-s。
def _extract_yahei_bytes() -> bytes | None:
    msyh = r"C:\Windows\Fonts\msyh.ttc"
    if not Path(msyh).exists():
        return None
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(msyh, fontNumber=0)
        buf = io.BytesIO()
        f.save(buf)
        return buf.getvalue()
    except Exception:
        logger.warning("提取微软雅黑 TTF 失败，回退内置字体", exc_info=True)
        return None


_YAHEI_BYTES = _extract_yahei_bytes()
_TEMP_FONT_FILE: str | None = None  # 提取的雅黑 TTF 落盘路径（insert_text 需 fontfile）


def _load_font() -> fitz.Font:
    return fitz.Font(fontbuffer=_YAHEI_BYTES) if _YAHEI_BYTES else fitz.Font("china-s")


def _ensure_font_file() -> str | None:
    """把提取的雅黑 TTF 写到 ai_temp（insert_text 需要文件路径），复用缓存文件。"""
    global _TEMP_FONT_FILE
    if _TEMP_FONT_FILE and Path(_TEMP_FONT_FILE).exists():
        return _TEMP_FONT_FILE
    if not _YAHEI_BYTES:
        return None
    p = Path(__file__).resolve().parents[3] / "ai_temp" / "msyh_font.ttf"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_YAHEI_BYTES)
        _TEMP_FONT_FILE = str(p)
        return _TEMP_FONT_FILE
    except Exception:
        logger.warning("写入雅黑字体文件失败，回退内置字体", exc_info=True)
        return None


def _insert(page, point, text: str, size: float, color) -> None:
    """统一插入文本：雅黑（标准 TrueType 嵌入）优先，回退 china-s。"""
    fp = _ensure_font_file()
    if fp:
        page.insert_text(point, text, fontsize=size, fontname="yahei", fontfile=fp, color=color)
    else:
        page.insert_text(point, text, fontsize=size, fontname="china-s", color=color)


# ── 字段定位表 ──
# (变量key, 页码0based, 锚文本, 模式, 字号, dx, dy, max_width)
# dx: 值起点相对锚文本右缘的偏移；dy: 值起点 y 相对锚矩形底缘的偏移（微调对齐）
# 模式: after / after_left / bracket
_FIELDS = [
    # 页1 抬头
    ("companyName",    0, "〔公司名称〕", "bracket", 18.0, 0, -3, 115),
    ("offerNo",        0, "编号：",      "after",    8.2, 2, -2, 45),
    ("issueDate",      0, "日期：",      "after",    8.2, 2, -2, 78),
    ("candidateName",  0, "尊敬的",      "after",   10.5, 6, -2, 60),
    # 页1 正文
    ("position",       0, "〔岗位名称〕", "bracket",  9.4, 0, -2, 60),
    # 页1 01 职位信息
    ("position",       0, "职位名称",    "after",    9.0, 80, -2, 70),
    ("department",     0, "所属部门",    "after",    9.0, 80, -2, 70),
    ("workLocation",   0, "工作地点",    "after",    9.0, 80, -2, 70),
    ("probationMonths", 0, "含试用期",   "after",    9.0, 3, -2, 40),
    ("expectedOnboardDate", 0, "预计入职日", "after", 9.0, 80, -2, 56),
    # 页1 02 薪酬
    ("baseSalary",     0, "税前人民币",  "after",    9.0, 3, -2, 110),
    ("performanceSalary", 0, "目标年薪的", "after",   9.0, 3, -2, 60),
    ("annualBonus",    0, "个月基本月薪", "after_left", 9.0, -22, -2, 18),
    ("allowance",      0, "合计约",      "after",    9.0, 3, -2, 100),
    # 页2 确认有效期 / 报到 / 试用期
    ("expiresAt",      1, "请在 ",       "after",    9.4, 3, -2, 96),
    ("expectedOnboardDate", 1, "报到时间", "after",   9.0, 80, -10, 55),
    ("workAddress",    1, "报到地址",    "after",    9.0, 80, -2, 85),
    ("probationMonths", 1, "试用期 ",    "after",    9.4, 3, -2, 16),
    # 页2 签署（用人单位〔公司名称〕）
    ("companyName",    1, "〔公司名称〕", "bracket",  7.9, 0, -2, 70),
    # 页3 签署
    ("candidateName",  2, "姓名：",      "after",    9.0, 3, -2, 95),
    ("signDate",       2, "签署日期：",  "after",    9.0, 3, -2, 90),
]


def build_offer_vars(offer, candidate, position, company_info: dict,
                     today: date | None = None) -> dict:
    """组装填充变量。无数据 → 空串（该字段留空不写入）。"""
    comp = (offer.compensation or {}) if offer else {}
    other = (offer.other_terms or {}) if offer else {}
    onboard = offer.expected_onboard_date if offer else None
    expires = None
    if offer:
        expires = offer.expires_at or getattr(offer, "token_expires_at", None)
    today = today or date.today()

    def fmt_date(d) -> str:
        return d.strftime("%Y年%m月%d日") if d else ""

    offer_no = ""
    if offer and getattr(offer, "id", None):
        offer_no = str(offer.id).replace("-", "")[:8].upper()

    company = company_info or {}
    return {
        "companyName": company.get("companyName", ""),
        "offerNo": offer_no,
        "issueDate": fmt_date(today),
        "signDate": fmt_date(today),
        "candidateName": candidate.name if candidate else "",
        "position": position.name if position else "",
        "department": (offer.department if offer else "") or (position.department if position else ""),
        "workLocation": (offer.work_location if offer else "") or "",
        "probationMonths": str(offer.probation_months or "") if offer else "",
        "expectedOnboardDate": fmt_date(onboard),
        "baseSalary": comp.get("base_salary", ""),
        "performanceSalary": comp.get("performance_salary", ""),
        "annualBonus": comp.get("annual_bonus_note", ""),
        "allowance": comp.get("allowance", ""),
        "reportMaterials": other.get("report_materials", ""),
        "expiresAt": fmt_date(expires),
        "workAddress": company.get("workAddress") or company.get("address", ""),
    }


def _fit_text(font: fitz.Font, text: str, size: float, max_width: float) -> tuple[str, float]:
    """按可用宽度自适应：优先缩字号，仍超则截断加省略号。"""
    if not max_width or max_width <= 0:
        return text, size
    if font.text_length(text, size) <= max_width:
        return text, size
    s = size
    while s > 6.0 and font.text_length(text, s) > max_width:
        s -= 0.5
    if font.text_length(text, s) <= max_width:
        return text, s
    cut = text
    while cut and font.text_length(cut + "…", s) > max_width:
        cut = cut[:-1]
    return cut + "…", s


def fill_template_pdf(data: dict) -> bytes:
    """打开模板 → 按字段表定位写入值 → 返回 PDF 字节。"""
    font = _load_font()
    doc = fitz.open(TEMPLATE_PATH)

    # 收集：bracket（先涂白再写黑字）；after / after_left（写蓝字）
    redact_items = {}   # pno -> [(rect, value, size, maxw)]
    after_items = []    # (pno, point, value, size, color)

    for key, pno, anchor, mode, size, dx, dy, maxw in _FIELDS:
        value = str(data.get(key, "")).strip()
        if not value:
            continue  # 无数据，保持原下划线
        page = doc[pno]
        rects = page.search_for(anchor)
        if not rects:
            logger.warning("Offer 模板字段锚文本未命中: %s (page %s)", anchor, pno + 1)
            continue
        rect = rects[0]
        if mode == "bracket":
            redact_items.setdefault(pno, []).append((rect, value, size, maxw))
        elif mode == "after_left":
            v, s = _fit_text(font, value, size, maxw)
            x = rect.x0 + dx - font.text_length(v, s)
            after_items.append((pno, (x, rect.y1 + dy), v, s, _FILL_COLOR))
        else:  # after
            v, s = _fit_text(font, value, size, maxw)
            after_items.append((pno, (rect.x1 + dx, rect.y1 + dy), v, s, _FILL_COLOR))

    # 1) 涂白〔〕占位符（bracket 区域）
    for pno, items in redact_items.items():
        page = doc[pno]
        for rect, _v, _s, _w in items:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()

    # 2) 写入值
    for pno, (x, y), value, size, color in after_items:
        _insert(doc[pno], (x, y), value, size, color)
    for pno, items in redact_items.items():
        page = doc[pno]
        for rect, value, size, maxw in items:
            v, s = _fit_text(font, value, size, maxw)
            y = rect.y1 + (-3 if size >= 12 else -2)
            _insert(page, (rect.x0, y), v, s, _ISSUE_COLOR)

    buf = io.BytesIO()
    doc.save(buf, garbage=3, deflate=True)
    doc.close()
    return buf.getvalue()
