"""权限点注册表 + 角色→权限种子映射。

命名规范 {模块}:{动作}。设计为可扩展 registry(二期+8/三期+9/四期+5),
新增权限只需往 PERMISSION_REGISTRY 追加,往 ROLE_PERMISSIONS 对应角色追加。

角色编码(原型口径,第四期追加 equity_committee):
  ceo         公司负责人
  hr          HR
  manager     部门负责人
  interviewer 面试官
  mentor      带教人
  project_lead 项目负责人
  employee    员工
  admin       系统管理员
"""
from __future__ import annotations

# ── 角色定义(code -> (name, description)) ──
ROLES: dict[str, tuple[str, str]] = {
    "ceo": ("公司负责人", "全局只读 + 晋级/期权终审"),
    "hr": ("HR", "招聘全流程 + 人事管理"),
    "manager": ("部门负责人", "招聘需求确认 + 录用/转正审批"),
    "interviewer": ("面试官", "简历查看 + 面试评分"),
    "mentor": ("带教人", "试用期带教 + 培训"),
    "project_lead": ("项目负责人", "工作任务 + 积分/奖惩"),
    "employee": ("员工", "员工自助"),
    "admin": ("系统管理员", "系统与权限管理"),
    # 第四期追加:
    "equity_committee": ("期权委员会", "期权审批联签"),
}

# ── 权限点注册表(可扩展) ──
# value = 人类可读说明,仅用于管理端展示。
PERMISSION_REGISTRY: dict[str, str] = {
    # 招聘需求
    "recruitment_request:create": "创建/编辑招聘需求",
    "recruitment_request:confirm": "确认招聘需求",
    "position:publish": "发布岗位",
    "position:manage": "岗位增删改",
    # 简历
    "resume:view": "查看简历",
    "resume:decide": "简历处置决策",
    # 面试
    "interview:manage": "安排面试/出题/结论",
    "interview:score": "面试评分",
    # 录用
    "offer:approve": "录用审批",
    # 敏感字段
    "salary:view": "查看薪资",
    "salary:manage": "调整薪资/奖金",
    "equity:view": "查看期权(四期)",
    # 系统
    "audit:view": "查看操作日志",
    "system:manage": "用户与系统管理",
    "llm:config": "LLM 配置",
    # 第二期: 试用期 + 培训带教
    "probation:manage": "试用期计划/任务管理",
    "probation:submit": "提交试用期任务",
    "probation:accept": "验收试用期任务",
    "confirmation:approve": "转正审批",
    "mentor:record": "填写带教记录",
    "training:view": "查看培训内容",
    "training:manage": "管理培训课程",
    "training:confirm": "确认培训完成",
    # 第三期: 任务积分
    "task:create": "创建/发布工作任务",
    "task:submit": "提交任务成果",
    "task:accept": "验收任务",
    "task:view": "查看任务列表",
    "points:confirm": "确认积分生效",
    "points:deduct": "扣分操作",
    "points:reward": "奖励操作",
    "self:appeal": "提交积分申诉",
    "self:confirm": "员工自助确认",
    "self:points:view": "查看本人积分",
    # 第四期: 人才池/晋级/期权
    "talent:view": "查看人才池",
    "talent:manage": "管理人才画像",
    "promotion:approve": "晋级审批",
    "equity:view": "查看期权",
    "equity:approve": "期权审批",
}

# ── 角色 -> 权限点集合(种子) ──
# 只读全局能力用具体权限点表达;通配由 security.require_permission 对 admin 处理。
_ALL_READ = ["resume:view", "audit:view"]

ROLE_PERMISSIONS: dict[str, list[str]] = {
    "ceo": _ALL_READ + [
        "offer:approve", "salary:view", "salary:manage", "equity:view",
        "talent:view", "promotion:approve", "equity:approve",  # P4
    ],
    "hr": [
        "recruitment_request:create", "recruitment_request:confirm",
        "position:publish", "position:manage",
        "resume:view", "resume:decide",
        "interview:manage",
        "salary:view", "salary:manage", "audit:view",
        "probation:manage", "training:view", "training:manage",
        "task:view", "points:confirm",
        "talent:view", "talent:manage", "equity:view",  # P4
    ],
    "manager": [
        "recruitment_request:create", "recruitment_request:confirm",
        "offer:approve", "interview:score", "salary:view",
        "confirmation:approve",
        "talent:view",  # P4
    ],
    "interviewer": ["resume:view", "interview:score"],
    "mentor": [
        "probation:manage", "probation:accept", "mentor:record",
        "training:view", "training:manage", "training:confirm",
    ],
    "project_lead": [
        "probation:accept",
        "task:create", "task:accept", "task:view",
        "points:confirm", "points:deduct", "points:reward",
    ],
    "employee": [
        "training:view", "training:confirm", "probation:submit",
        "task:submit", "task:view",
        "self:appeal", "self:confirm", "self:points:view",
    ],
    "admin": ["system:manage", "audit:view", "llm:config"],
    "equity_committee": ["equity:view", "equity:approve", "talent:view"],
}

# admin/ceo 的通配读:security.require_permission 中,持 system:manage 视为放行一切。
WILDCARD_PERMISSION = "system:manage"


def permissions_for_roles(role_codes: list[str]) -> set[str]:
    """把用户所有角色的权限点求并集。"""
    perms: set[str] = set()
    for rc in role_codes:
        perms.update(ROLE_PERMISSIONS.get(rc, []))
    return perms


# ────────────────────────────────────────────────────────────
# AI 对话能力描述（按权限点裁剪 system prompt 用）
# ────────────────────────────────────────────────────────────
# 背景：对话助手 system.txt 里曾硬编码"你可以完成以下所有功能"，把 9 大模块
# 全列出来，导致 HR 也会被 AI 告知"能做系统设置/绩效/知识库"等本无权限的事，
# 与工具层（app/agent/tools.py 的 TOOL_PERMISSIONS 按 current_user 裁剪）严重脱节。
#
# 本模块与工具层同源：每个「能力段」声明所需权限点（任一命中即可），
# AI 的可见能力据此收敛到用户真实权限，不再宣称无权功能。
#
# 语义约定（与 CurrentUser.has() 一致）：admin 持 WILDCARD_PERMISSION
# 通配放行，任何能力段都会命中，不另做特殊分支——admin 本身就是全权限。
#
# 判定原则（避免只读权限误扩整模块）：
#   能力段用「该模块的核心写/操作权限点」判定——只有用户真能**操作**这个模块
#   时才展示「XX管理」，绝不因挂了一只读权限（如 audit:view / equity:view）
#   就把整个管理模块宣称给低权限角色（例：HR 有 audit:view 但无 system:manage，
#   不能因此宣称"可管理系统配置"）。

# 能力段：(展示名, 任一命中即可, 能力描述)
_CAPABILITY_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "简历筛选",
        ("resume:view", "resume:decide"),
        "查询/查看候选人简历与匹配分，按岗位/状态/关键词找人，查看同岗位排名，"
        "删除或重新解析简历",
    ),
    (
        "岗位管理",
        ("position:manage",),
        "创建/更新/删除招聘岗位，管理 JD 与岗位题库",
    ),
    (
        "面试出题",
        ("interview:manage", "interview:score"),
        "查看候选人题单与岗位通用题库；持面试管理权限者可生成/保存/替换一面二面题目",
    ),
    (
        "面试评定",
        ("interview:score",),
        "录入候选人回答、AI 评分、查看排行榜并提交面试结论",
    ),
    (
        "试用期考核",
        ("probation:manage", "probation:accept", "probation:submit"),
        "查看试用期员工与周任务；按权限新增/跟踪周任务、AI 自动或手动评估",
    ),
    (
        "转正审批",
        ("confirmation:approve",),
        "查看待转正员工并处理转正审批",
    ),
    (
        "工作任务与积分",
        ("task:create", "task:submit", "task:accept"),
        "发布/提交/验收工作任务，处理积分确认与奖惩",
    ),
    (
        "培训管理",
        ("training:manage", "training:confirm"),
        "查看/管理培训课程与进度，确认培训完成",
    ),
    (
        "绩效管理",
        ("talent:manage", "salary:view", "salary:manage"),
        "查看绩效统计/部门绩效/等级分布，发起季度考核与调整奖金",
    ),
    (
        "录用审批",
        ("offer:approve",),
        "处理录用审批",
    ),
    (
        "人才池与晋升",
        ("talent:manage", "promotion:approve"),
        "查看/管理人才画像，处理晋级审批",
    ),
    (
        "期权审批",
        ("equity:approve",),
        "处理期权候选人审批与联签",
    ),
    (
        "系统管理",
        ("system:manage", "llm:config"),
        "修改系统配置、管理用户与角色",
    ),
    (
        "操作日志",
        ("audit:view",),
        "查看审计操作日志",
    ),
)


def build_capability_lines(permissions: set[str]) -> list[str]:
    """按权限点并集生成 AI 可见的业务能力描述行。

    持 WILDCARD_PERMISSION（admin）视为通配，渲染全部能力段。
    否则仅渲染「所需权限中任一命中」的能力段。
    """
    wildcard = WILDCARD_PERMISSION in permissions
    lines: list[str] = []
    for label, requires, desc in _CAPABILITY_RULES:
        if wildcard or any(r in permissions for r in requires):
            lines.append(f"- **{label}**：{desc}")
    return lines
