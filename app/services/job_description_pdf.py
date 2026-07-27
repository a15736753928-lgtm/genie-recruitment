"""
岗位说明书 PDF 生成服务

使用 reportlab 生成正式的、可签字的岗位说明书 PDF 文档。
支持中文（思源黑体/微软雅黑），包含：
- 公司 logo 占位
- 结构化内容展示
- 审批签字栏
"""
import io
import os
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)

# ── 中文字体注册 ──────────────────────────────────────────────

_FONT_REGISTERED = False
_FONT_NAME = "Helvetica"  # 默认 fallback
_FONT_NAME_BOLD = "Helvetica-Bold"


def _register_chinese_fonts():
    """尝试注册中文字体（微软雅黑 / 思源黑体）"""
    global _FONT_REGISTERED, _FONT_NAME, _FONT_NAME_BOLD
    if _FONT_REGISTERED:
        return

    # 常见中文字体路径
    font_candidates = [
        # Windows 系统字体
        ("msyh", "C:/Windows/Fonts/msyh.ttc"),
        ("msyhbd", "C:/Windows/Fonts/msyhbd.ttc"),
        ("simhei", "C:/Windows/Fonts/simhei.ttf"),
        ("simsun", "C:/Windows/Fonts/simsun.ttc"),
    ]

    for name, path in font_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                if name in ("msyh", "simhei"):
                    _FONT_NAME = name
                if name in ("msyhbd", "simhei"):
                    _FONT_NAME_BOLD = name
                _FONT_REGISTERED = True
            except Exception:
                continue

    # 如果没找到系统字体，尝试使用内置的 CJK 字体
    if not _FONT_REGISTERED:
        try:
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            _FONT_NAME = "STSong-Light"
            _FONT_NAME_BOLD = "STSong-Light"
            _FONT_REGISTERED = True
        except Exception:
            pass


# ── 样式定义 ──────────────────────────────────────────────────


def _get_styles():
    """获取文档样式"""
    _register_chinese_fonts()

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            "DocTitle",
            fontName=_FONT_NAME_BOLD,
            fontSize=18,
            leading=24,
            alignment=1,  # 居中
            spaceAfter=6 * mm,
            textColor=colors.HexColor("#1e293b"),
        )
    )
    styles.add(
        ParagraphStyle(
            "DocSubtitle",
            fontName=_FONT_NAME,
            fontSize=10,
            leading=14,
            alignment=1,
            spaceAfter=8 * mm,
            textColor=colors.HexColor("#64748b"),
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionTitle",
            fontName=_FONT_NAME_BOLD,
            fontSize=12,
            leading=16,
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
            textColor=colors.HexColor("#2563eb"),
            borderPadding=(0, 0, 2, 0),
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyText_CN",
            fontName=_FONT_NAME,
            fontSize=10,
            leading=16,
            spaceAfter=2 * mm,
            textColor=colors.HexColor("#334155"),
        )
    )
    styles.add(
        ParagraphStyle(
            "BulletItem",
            fontName=_FONT_NAME,
            fontSize=10,
            leading=15,
            leftIndent=12 * mm,
            bulletIndent=6 * mm,
            spaceAfter=1 * mm,
            textColor=colors.HexColor("#334155"),
        )
    )
    styles.add(
        ParagraphStyle(
            "TableCell",
            fontName=_FONT_NAME,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
        )
    )
    styles.add(
        ParagraphStyle(
            "TableLabel",
            fontName=_FONT_NAME_BOLD,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#475569"),
        )
    )
    styles.add(
        ParagraphStyle(
            "SignatureLabel",
            fontName=_FONT_NAME,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
        )
    )

    return styles


# ── 辅助函数 ──────────────────────────────────────────────────


def _section_title(text: str) -> str:
    """格式化章节标题（带蓝色竖线装饰）"""
    return f'<font color="#2563eb">■</font>  {text}'


def _build_info_table(
    data: Dict[str, str], styles, col_widths: Optional[list] = None
) -> Table:
    """构建信息表格"""
    if col_widths is None:
        col_widths = [35 * mm, 50 * mm, 35 * mm, 50 * mm]

    table_data = []
    keys = list(data.keys())
    for i in range(0, len(keys), 2):
        row = []
        for j in range(2):
            idx = i + j
            if idx < len(keys):
                k = keys[idx]
                v = data[k]
                row.append(Paragraph(f"<b>{k}</b>", styles["TableLabel"]))
                row.append(Paragraph(str(v), styles["TableCell"]))
            else:
                row.extend(["", ""])
        table_data.append(row)

    table = Table(table_data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f8fafc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _build_bullet_list(items: list, styles) -> list:
    """构建带项目符号的列表"""
    elements = []
    for i, item in enumerate(items, 1):
        elements.append(
            Paragraph(f"• {item}", styles["BulletItem"])
        )
    return elements


def _build_signature_block(styles) -> Table:
    """构建审批签字栏"""
    sig_data = [
        [
            Paragraph("<b>HR 签字：</b>", styles["TableLabel"]),
            Paragraph("____________________", styles["TableCell"]),
            Paragraph("<b>日期：</b>", styles["TableLabel"]),
            Paragraph("____年____月____日", styles["TableCell"]),
        ],
        [
            Paragraph("<b>部门负责人签字：</b>", styles["TableLabel"]),
            Paragraph("____________________", styles["TableCell"]),
            Paragraph("<b>日期：</b>", styles["TableLabel"]),
            Paragraph("____年____月____日", styles["TableCell"]),
        ],
        [
            Paragraph("<b>员工签字：</b>", styles["TableLabel"]),
            Paragraph("____________________", styles["TableCell"]),
            Paragraph("<b>日期：</b>", styles["TableLabel"]),
            Paragraph("____年____月____日", styles["TableCell"]),
        ],
    ]

    table = Table(sig_data, colWidths=[35 * mm, 55 * mm, 25 * mm, 55 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f8fafc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


# ── 主入口 ────────────────────────────────────────────────────


def generate_job_description_pdf(
    job_data: Dict[str, Any],
    output_path: Optional[str] = None,
) -> bytes:
    """
    生成岗位说明书 PDF

    Args:
        job_data: 结构化岗位说明书数据
        output_path: 可选，输出文件路径；为 None 时返回 bytes

    Returns:
        PDF 文件的 bytes（output_path 为 None 时）
    """
    _register_chinese_fonts()
    styles = _get_styles()

    # 创建文档
    if output_path:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=25 * mm,
            bottomMargin=25 * mm,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
        )
    else:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=25 * mm,
            bottomMargin=25 * mm,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
        )

    elements = []

    # ── 标题区 ──────────────────────────────────────────────
    elements.append(Paragraph("岗 位 说 明 书", styles["DocTitle"]))
    elements.append(
        Paragraph(
            f'岗位名称：{job_data.get("basicInfo", {}).get("positionName", "—")}',
            styles["DocSubtitle"],
        )
    )
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb")))
    elements.append(Spacer(1, 4 * mm))

    # ── 一、岗位基础信息 ──────────────────────────────────
    elements.append(Paragraph(_section_title("一、岗位基础信息"), styles["SectionTitle"]))
    basic = job_data.get("basicInfo", {})
    info_table = _build_info_table(
        {
            "岗位名称": basic.get("positionName", "—"),
            "所属部门": basic.get("department", "—"),
            "岗位编制": f'{basic.get("headcount", "—")} 人',
            "汇报对象": basic.get("reportTo", "—"),
            "薪资范围": basic.get("salaryRange", "—"),
            "工作地点": basic.get("workLocation", "—"),
            "用工性质": basic.get("employmentType", "—"),
            "岗位等级": basic.get("level", "—"),
        },
        styles,
    )
    elements.append(info_table)

    # ── 二、岗位使命 / 工作目标 ────────────────────────────
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(_section_title("二、岗位使命 / 工作目标"), styles["SectionTitle"]))
    mission = job_data.get("mission", "—")
    elements.append(Paragraph(mission, styles["BodyText_CN"]))

    # ── 三、主要工作职责 ──────────────────────────────────
    elements.append(Paragraph(_section_title("三、主要工作职责"), styles["SectionTitle"]))
    responsibilities = job_data.get("responsibilities", [])
    if responsibilities:
        elements.extend(_build_bullet_list(responsibilities, styles))
    else:
        elements.append(Paragraph("—", styles["BodyText_CN"]))

    # ── 四、任职资格要求 ──────────────────────────────────
    elements.append(Paragraph(_section_title("四、任职资格要求"), styles["SectionTitle"]))
    quals = job_data.get("qualifications", {})

    elements.append(Paragraph("<b>必须具备：</b>", styles["BodyText_CN"]))
    required = quals.get("required", [])
    if required:
        elements.extend(_build_bullet_list(required, styles))

    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph("<b>优先条件：</b>", styles["BodyText_CN"]))
    preferred = quals.get("preferred", [])
    if preferred:
        elements.extend(_build_bullet_list(preferred, styles))

    # ── 五、岗位工作权限 ──────────────────────────────────
    elements.append(Paragraph(_section_title("五、岗位工作权限"), styles["SectionTitle"]))
    permissions = job_data.get("permissions", [])
    if permissions:
        elements.extend(_build_bullet_list(permissions, styles))
    else:
        elements.append(Paragraph("—", styles["BodyText_CN"]))

    # ── 六、内外部协作关系 ────────────────────────────────
    elements.append(Paragraph(_section_title("六、内外部协作关系"), styles["SectionTitle"]))
    collabs = job_data.get("collaborations", {})

    elements.append(Paragraph("<b>内部协作：</b>", styles["BodyText_CN"]))
    internal = collabs.get("internal", [])
    if internal:
        elements.extend(_build_bullet_list(internal, styles))

    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph("<b>外部协作：</b>", styles["BodyText_CN"]))
    external = collabs.get("external", [])
    if external:
        elements.extend(_build_bullet_list(external, styles))

    # ── 七、工作环境、工时、出差 ──────────────────────────
    elements.append(
        Paragraph(_section_title("七、工作环境、工时、出差情况"), styles["SectionTitle"])
    )
    work_env = job_data.get("workEnvironment", {})
    env_table = _build_info_table(
        {
            "办公环境": work_env.get("officeType", "—"),
            "工作时间": work_env.get("workingHours", "—"),
            "加班情况": work_env.get("overtime", "—"),
            "出差要求": work_env.get("travel", "—"),
        },
        styles,
        col_widths=[30 * mm, 55 * mm, 30 * mm, 55 * mm],
    )
    elements.append(env_table)

    # ── 八、核心绩效考核指标 ──────────────────────────────
    elements.append(
        Paragraph(_section_title("八、核心绩效考核指标"), styles["SectionTitle"])
    )
    kpi = job_data.get("kpi", [])
    if kpi:
        elements.extend(_build_bullet_list(kpi, styles))
    else:
        elements.append(Paragraph("—", styles["BodyText_CN"]))

    # ── 九、职业发展通道 ──────────────────────────────────
    elements.append(Paragraph(_section_title("九、职业发展通道"), styles["SectionTitle"]))
    career = job_data.get("careerPath", "—")
    elements.append(Paragraph(career, styles["BodyText_CN"]))

    # ── 十、审批签字栏 ────────────────────────────────────
    elements.append(Spacer(1, 8 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8")))
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(_section_title("十、审批签字栏"), styles["SectionTitle"]))
    elements.append(_build_signature_block(styles))

    # ── 页脚说明 ──────────────────────────────────────────
    elements.append(Spacer(1, 10 * mm))
    elements.append(
        Paragraph(
            "本岗位说明书经双方确认后生效，作为劳动合同附件。",
            ParagraphStyle(
                "Footer",
                fontName=_FONT_NAME,
                fontSize=8,
                leading=12,
                alignment=1,
                textColor=colors.HexColor("#94a3b8"),
            ),
        )
    )

    # 生成 PDF
    doc.build(elements)

    if output_path:
        return b""
    else:
        buffer.seek(0)
        return buffer.read()


# ── 便捷函数 ──────────────────────────────────────────────────


def generate_pdf_to_file(job_data: Dict[str, Any], output_path: str) -> str:
    """生成 PDF 并保存到文件，返回文件路径"""
    generate_job_description_pdf(job_data, output_path=output_path)
    return output_path


def generate_pdf_bytes(job_data: Dict[str, Any]) -> bytes:
    """生成 PDF 并返回 bytes"""
    return generate_job_description_pdf(job_data)
