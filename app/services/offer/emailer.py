"""Offer 邮件发送 —— 一键把确认链接发到候选人邮箱。

SMTP 走 Python 标准库 smtplib/email（零新依赖），参数从 system_settings 读取：
  smtpHost / smtpPort / smtpUseSsl / smtpUser / smtpPassword / smtpFrom
未配置 SMTP 或发送失败 → 降级为 log_only（沿用 notification.py 约定），
返回 emailSent=false，由前端提示 HR 复制 confirmUrl 手动发送。
"""
from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

from app.services.system.system_settings import get_system_setting

logger = logging.getLogger("genie.offer.email")

SMTP_MAX_AGE = 15  # 秒


async def send_offer_email(db, offer, candidate, confirm_url: str) -> tuple[bool, str]:
    """向候选人发送 Offer 确认邮件。

    返回 (email_sent: bool, message: str)。message 为失败原因或成功简述。
    调用方负责在 status=sent 后调用；candidate.email 为空则直接降级。
    """
    from app.services.system.notification import notify_if

    if not candidate or not candidate.email:
        await notify_if(db, "notifyOfferPending", "offer_sent",
                        f"候选人 {candidate.name if candidate else ''} 无邮箱，Offer 链接需人工复制发送")
        return False, "候选人无邮箱，已生成链接，请复制手动发送"

    host = str(await get_system_setting(db, "smtpHost", "") or "").strip()
    if not host:
        await notify_if(db, "notifyOfferPending", "offer_sent",
                        f"SMTP 未配置，Offer 链接已生成待复制：{candidate.name} → {candidate.email}")
        return False, "SMTP 未配置，已生成链接，请复制手动发送"

    port = int(await get_system_setting(db, "smtpPort", 465) or 465)
    use_ssl = bool(await get_system_setting(db, "smtpUseSsl", True))
    smtp_user = str(await get_system_setting(db, "smtpUser", "") or "").strip()
    smtp_pass = str(await get_system_setting(db, "smtpPassword", "") or "").strip()
    smtp_from = str(await get_system_setting(db, "smtpFrom", "") or "").strip() or smtp_user
    company = str(await get_system_setting(db, "companyName", "公司") or "")

    if not smtp_from:
        await notify_if(db, "notifyOfferPending", "offer_sent",
                        f"SMTP 发件人未配置，Offer 链接已生成待复制：{candidate.name}")
        return False, "SMTP 发件人未配置，已生成链接，请复制手动发送"

    try:
        # 发送在 asyncio 事件循环内会阻塞 —— 用 to_thread 避免卡住 SSE/请求
        import asyncio
        ok, msg = await asyncio.to_thread(
            _smtp_send, host, port, use_ssl, smtp_user, smtp_pass,
            smtp_from, company, candidate, confirm_url, offer,
        )
        if ok:
            await notify_if(db, "notifyOfferPending", "offer_sent",
                            f"Offer 已邮件发送：{candidate.name} → {candidate.email}", {"url": confirm_url})
        else:
            await notify_if(db, "notifyOfferPending", "offer_sent",
                            f"Offer 邮件发送失败({msg})，链接已生成待复制：{candidate.name}")
        return ok, msg
    except Exception as e:  # noqa: BLE001 —— 发信失败绝不能让发送端点 500
        logger.warning("Offer 邮件发送异常: %s", e)
        await notify_if(db, "notifyOfferPending", "offer_sent",
                        f"Offer 邮件发送异常，链接已生成待复制：{candidate.name}")
        return False, "邮件发送异常，已生成链接，请复制手动发送"


def _smtp_send(
    host: str, port: int, use_ssl: bool, user: str, password: str,
    from_addr: str, company: str, candidate, confirm_url: str, offer,
) -> tuple[bool, str]:
    """同步阻塞的 SMTP 发送。返回 (ok, message)。"""
    subject = f"{company} —— 录用通知（Offer 确认）"
    html = _build_email_html(company, candidate, confirm_url, offer)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{company} <{from_addr}>"
    msg["To"] = candidate.email
    msg.attach(MIMEText(html, "html", "utf-8"))

    if use_ssl:
        server: smtplib.SMTP = smtplib.SMTP_SSL(host, port, timeout=SMTP_MAX_AGE)
    else:
        server = smtplib.SMTP(host, port, timeout=SMTP_MAX_AGE)
        server.ehlo()
        server.starttls()
        server.ehlo()
    try:
        if user:
            server.login(user, password)
        server.sendmail(from_addr, [candidate.email], msg.as_string())
    finally:
        try:
            server.quit()
        except Exception:  # noqa: BLE001
            pass
    return True, "已发送"


def _build_email_html(company: str, candidate, confirm_url: str, offer) -> str:
    """候选人邮箱正文 —— 简述 + 确认链接 + 有效期提示。"""
    validity = offer.validity_days or 7
    return f"""<!doctype html><html><body style="font-family:'Microsoft YaHei',Arial,sans-serif;background:#f5f7fa;margin:0;padding:24px;">
<div style="max-width:560px;margin:0 auto;background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;">
  <div style="background:#2563eb;padding:20px 28px;color:#fff;">
    <h2 style="margin:0;font-size:18px;">{company} · 录用通知</h2>
  </div>
  <div style="padding:24px 28px;color:#1e293b;font-size:14px;line-height:1.8;">
    <p>{candidate.name} 您好：</p>
    <p>恭喜您通过了我司全部面试流程，我们诚挚邀请您加入 {company}。</p>
    <p>请点击下方按钮在线查看您的 <b>Offer 详情</b>，并在 <b>{validity} 天内</b>完成确认；<span style="color:#dc2626;">逾期未操作将视为放弃。</span></p>
    <p style="text-align:center;margin:28px 0;">
      <a href="{confirm_url}" style="display:inline-block;background:#2563eb;color:#fff;padding:12px 32px;border-radius:8px;text-decoration:none;font-weight:600;">查看 Offer 并确认</a>
    </p>
    <p style="color:#64748b;font-size:12px;">若按钮无法点击，请复制以下链接到浏览器打开：<br/><a href="{confirm_url}" style="color:#2563eb;word-break:break-all;">{confirm_url}</a></p>
    <p style="color:#64748b;font-size:12px;">本链接仅限您本人使用，请勿转发。</p>
  </div>
</div></body></html>"""
