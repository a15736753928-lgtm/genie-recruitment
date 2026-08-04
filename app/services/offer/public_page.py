"""候选人线上确认页 —— 后端渲染的独立 HTML（公开 token 路由，无需登录）。

候选人自己看自己的 Offer，**明文展示薪酬**（不走 salary:view 掩码）。
单文件内联 CSS，移动端友好，无外部资源。
"""
from __future__ import annotations

from typing import Any, Optional
from html import escape


def _esc(v: Any) -> str:
    return escape(str(v if v is not None else ""), quote=True)


def _fmt_date(v: Any) -> str:
    if not v:
        return "—"
    s = str(v)
    # datetime / date 序列化为 ISO，取前 10 位日期
    return s[:10]


def _money(v: Any) -> str:
    return _esc(v) if v is not None and v != "" else "—"


def render_confirm_page(ctx: dict[str, Any]) -> str:
    """Offer 待确认页（status=sent，候选人可接受/拒绝）。"""
    comp = ctx.get("compensation") or {}
    terms = ctx.get("employment_terms") or {}
    other = ctx.get("other_terms") or {}

    rows = ""
    if comp.get("base_salary") is not None and comp.get("base_salary") != "":
        rows += _row("基本工资", _money(comp.get("base_salary")))
    if comp.get("performance_salary") is not None and comp.get("performance_salary") != "":
        rows += _row("绩效工资", _money(comp.get("performance_salary")))
    if comp.get("allowance") is not None and comp.get("allowance") != "":
        rows += _row("补贴（餐补/交通等）", _money(comp.get("allowance")))
    if comp.get("annual_bonus_note"):
        rows += _row("年终奖说明", _esc(comp.get("annual_bonus_note")))
    if comp.get("salary_tax_flag"):
        rows += _row("薪资口径", "税前" if comp.get("salary_tax_flag") == "pre" else "税后")
    if comp.get("social_security_base"):
        rows += _row("社保公积金缴纳基数", _esc(comp.get("social_security_base")))
    if comp.get("social_security_start_month"):
        rows += _row("社保缴纳起始", _esc(comp.get("social_security_start_month")))

    contract_type = "固定期限" if terms.get("contract_type") == "fixed" else ("无固定期限" if terms.get("contract_type") == "non_fixed" else "—")
    work_mode = "全职" if terms.get("work_mode") == "fulltime" else ("外包" if terms.get("work_mode") == "outsourcing" else "—")

    confirm_url = ctx.get("confirm_url", "")
    accept_url = ctx.get("accept_url", "")
    decline_url = ctx.get("decline_url", "")

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Offer 确认 · {_esc(ctx.get('company'))}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Microsoft YaHei','PingFang SC',Arial,sans-serif; background: #f1f5f9; color: #1e293b; padding: 24px 16px; }}
  .card {{ max-width: 640px; margin: 0 auto; background: #fff; border-radius: 14px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,.06); }}
  .head {{ background: #2563eb; color: #fff; padding: 24px 28px; }}
  .head h1 {{ font-size: 20px; margin-bottom: 6px; }}
  .head p {{ font-size: 13px; opacity: .92; }}
  .body {{ padding: 24px 28px; }}
  .section {{ margin-bottom: 22px; }}
  .section h2 {{ font-size: 14px; color: #2563eb; border-left: 4px solid #2563eb; padding-left: 10px; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  td {{ padding: 8px 6px; border-bottom: 1px solid #f1f5f9; }}
  td:first-child {{ color: #64748b; width: 40%; }}
  .terms {{ font-size: 13px; color: #475569; line-height: 1.9; white-space: pre-wrap; }}
  .notice {{ background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; border-radius: 10px; padding: 12px 14px; font-size: 13px; margin-bottom: 22px; }}
  .actions {{ text-align: center; padding-bottom: 8px; }}
  .btn {{ display: inline-block; padding: 13px 42px; border-radius: 9px; font-size: 15px; font-weight: 600; text-decoration: none; border: none; cursor: pointer; font-family: inherit; }}
  .btn-accept {{ background: #16a34a; color: #fff; }}
  .btn-decline {{ background: #fff; color: #dc2626; border: 1px solid #fecaca; margin-left: 12px; }}
  .reason {{ margin: 14px auto 0; max-width: 420px; text-align: left; }}
  .reason textarea {{ width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; font-size: 13px; font-family: inherit; resize: vertical; min-height: 70px; }}
  .foot {{ text-align: center; color: #94a3b8; font-size: 12px; padding: 16px; border-top: 1px solid #f1f5f9; }}
</style></head><body>
<div class="card">
  <div class="head">
    <h1>录用通知 · {_esc(ctx.get('company'))}</h1>
    <p>Offer 有效期至 <b>{_esc(ctx.get('token_expires_at'))}</b>，逾期未操作将自动视为放弃。</p>
  </div>
  <div class="body">
    <div class="section">
      <h2>候选人信息</h2>
      <table>
        {_row("姓名", _esc(ctx.get('candidate_name')))}
        {_row("联系电话", _esc(ctx.get('candidate_phone')))}
        {_row("邮箱", _esc(ctx.get('candidate_email')))}
        {_row("应聘岗位", _esc(ctx.get('position_name')))}
        {_row("所属部门", _esc(ctx.get('department')))}
        {_row("工作地点", _esc(ctx.get('work_location')))}
        {_row("预计入职日期", _fmt_date(ctx.get('expected_onboard_date')))}
        {_row("试用期", f"{_esc(ctx.get('probation_months'))} 个月" if ctx.get('probation_months') else "—")}
      </table>
    </div>
    <div class="section">
      <h2>薪酬福利</h2>
      <table>{rows or _empty_row("薪酬信息详见正式 Offer")}</table>
    </div>
    <div class="section">
      <h2>岗位约定</h2>
      <table>
        {_row("劳动合同类型", contract_type)}
        {_row("工作模式", work_mode)}
      </table>
    </div>
    <div class="section">
      <h2>其他条款</h2>
      <div class="terms">{_terms_block(other)}</div>
    </div>
    <div class="notice">请在有效期内点击下方按钮确认。同意后 HR 将尽快与您办理入职手续；拒绝请填写原因，我们尊重您的决定。</div>
    <div class="actions">
      <form method="post" action="{_esc(accept_url)}" style="display:inline-block;">
        <button type="submit" class="btn btn-accept">同意并接受</button>
      </form>
      <span id="declineToggle" class="btn btn-decline">拒绝 Offer</span>
      <div id="declinePanel" class="reason" style="display:none;">
        <textarea id="declineReason" name="reason" placeholder="请填写拒绝原因（可选）"></textarea>
        <p style="margin-top:10px;text-align:center;">
          <button type="button" id="declineConfirm" class="btn btn-decline" style="margin-left:0;">确认拒绝</button>
        </p>
      </div>
    </div>
  </div>
  <div class="foot">本链接仅限您本人使用，请勿转发。如有疑问请联系 HR。</div>
</div>
<script>
  var toggle = document.getElementById('declineToggle');
  var panel = document.getElementById('declinePanel');
  var reason = document.getElementById('declineReason');
  var confirmBtn = document.getElementById('declineConfirm');
  toggle.addEventListener('click', function () {{
    var show = panel.style.display !== 'block';
    panel.style.display = show ? 'block' : 'none';
    if (show) reason.focus();
  }});
  confirmBtn.addEventListener('click', function () {{
    var f = document.createElement('form');
    f.method = 'post';
    f.action = '{_esc(decline_url)}';
    var input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'reason';
    input.value = reason.value;
    f.appendChild(input);
    document.body.appendChild(f);
    f.submit();
  }});
</script>
</body></html>"""


def _terms_block(other: dict[str, Any]) -> str:
    lines = []
    if other.get("report_materials"):
        lines.append(f"报到所需材料：{_esc(other['report_materials'])}")
    if other.get("probation_requirements"):
        lines.append(f"试用期要求：{_esc(other['probation_requirements'])}")
    if other.get("non_compete_nda"):
        lines.append(f"竞业/保密提示：{_esc(other['non_compete_nda'])}")
    if other.get("remark"):
        lines.append(_esc(other["remark"]))
    return "<br/>".join(lines) if lines else "详见正式 Offer"


def render_result_page(state: str, ctx: dict[str, Any], message: str = "") -> str:
    """结果页：accepted / declined / expired / voided / not_found / already_processed。"""
    titles = {
        "accepted": "已确认接受",
        "declined": "已记录拒绝",
        "expired": "Offer 已过期",
        "voided": "Offer 已失效",
        "not_found": "链接无效",
        "already_processed": "该 Offer 已处理",
    }
    colors = {
        "accepted": "#16a34a", "declined": "#dc2626", "expired": "#d97706",
        "voided": "#64748b", "not_found": "#dc2626", "already_processed": "#64748b",
    }
    icons = {
        "accepted": "✓", "declined": "✕", "expired": "⏱", "voided": "—",
        "not_found": "!", "already_processed": "·",
    }
    title = titles.get(state, "Offer")
    color = colors.get(state, "#2563eb")
    icon = icons.get(state, "·")
    body_text = message or {
        "accepted": f"{_esc(ctx.get('candidate_name'))} 您好，您已成功接受 Offer。HR 将尽快与您联系办理入职手续。",
        "declined": "已记录您的反馈，感谢您对公司的关注，祝您前程似锦。",
        "expired": "该 Offer 已超过有效期，自动视为放弃。如需恢复请联系 HR。",
        "voided": "该 Offer 已失效，不再接受确认。请联系 HR 获取最新进展。",
        "not_found": "链接无效或不存在，请确认链接是否完整。",
        "already_processed": "该 Offer 已完成确认处理，请勿重复操作。",
    }.get(state, "详情请联系 HR。")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Offer</title>
<style>
  body {{ font-family:'Microsoft YaHei',Arial,sans-serif; background:#f1f5f9; display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; padding:24px; }}
  .card {{ max-width:420px; width:100%; background:#fff; border-radius:14px; padding:40px 32px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,.06); }}
  .icon {{ width:64px; height:64px; border-radius:50%; background:{color}14; color:{color}; font-size:30px; line-height:64px; margin:0 auto 18px; font-weight:700; }}
  h1 {{ font-size:18px; color:#1e293b; margin:0 0 12px; }}
  p {{ font-size:14px; color:#64748b; line-height:1.8; margin:0; }}
</style></head><body>
<div class="card"><div class="icon">{icon}</div><h1>{title}</h1><p>{body_text}</p></div>
</body></html>"""


def _row(label: str, value: str) -> str:
    return f"<tr><td>{label}</td><td><b>{value}</b></td></tr>"


def _empty_row(text: str) -> str:
    return f"<tr><td colspan='2' style='color:#94a3b8;'>{text}</td></tr>"
