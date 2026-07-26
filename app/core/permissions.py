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
    "media:upload": "上传面试录音录像",
    # 录用
    "offer:approve": "录用审批",
    # 敏感字段
    "salary:view": "查看薪资",
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
        "offer:approve", "salary:view", "equity:view",
        "talent:view", "promotion:approve", "equity:approve",  # P4
    ],
    "hr": [
        "recruitment_request:create", "recruitment_request:confirm",
        "position:publish", "position:manage",
        "resume:view", "resume:decide",
        "interview:manage",
        "salary:view", "audit:view",
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
    "interviewer": ["resume:view", "interview:score", "media:upload"],
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
