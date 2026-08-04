from app.models.recruitment import (
    Position, Candidate, CandidateSkill, CandidateEducation,
    CandidateWorkExperience, CandidateProjectExperience, CandidateAIAnalysis
)
from app.models.interview import (
    InterviewQuestion, InterviewEvaluation, InterviewTranscript, InterviewSegmentEvaluation
)
from app.models.probation import Employee, ProbationTask
from app.models.performance import PerformanceRecord, PerformanceQuarter
from app.models.knowledge import (
    KnowledgeCategory, KnowledgeItem
)
from app.models.agent_session import (
    AgentProject, AgentSession, AgentMessage, AgentMaterial, AgentTask
)
from app.models.settings import SystemSetting, AuditLog
# P0 基础设施
from app.models.auth import User, Role, UserRole, RolePermission
from app.models.system import StateTransition, ExceptionQueue
# 第一期业务表
from app.models.phase1 import (
    RecruitmentRequest, PositionCompetency,
    ResumeScore,
    Interview, InterviewerScore, AIInterviewReport,
    OfferApproval,
)
# Offer 管理模块附属表
from app.models.offer_module import (
    OfferTemplate, OfferApprovalFlow, OfferApprovalRecord, OfferAttachment,
)
# 第一期 AI 产物表
from app.models.phase1_ai import PositionAIArtifact
# 第二期业务表
from app.models.phase2 import (
    ProbationPlan, ProbationWeekReview, ConfirmationReview,
    TrainingCourse, EmployeeTrainingProgress, MentorRecord,
)
# 第三期业务表
from app.models.phase3 import (
    WorkTask, TaskAcceptance, PointRecord, RewardPenaltyRecord, Appeal,
)
# 第四期业务表
from app.models.phase4 import (
    TalentProfile, AbilityTag, PromotionRecord, EquityRecord,
    Project, ProjectAssignment,
)

__all__ = [
    "Position", "Candidate", "CandidateSkill", "CandidateEducation",
    "CandidateWorkExperience", "CandidateProjectExperience", "CandidateAIAnalysis",
    "InterviewQuestion", "InterviewEvaluation", "InterviewTranscript", "InterviewSegmentEvaluation",
    "Employee", "ProbationTask",
    "PerformanceRecord", "PerformanceQuarter",
    "KnowledgeCategory", "KnowledgeItem",
    "AgentProject", "AgentSession", "AgentMessage", "AgentMaterial", "AgentTask",
    "SystemSetting", "AuditLog",
    "User", "Role", "UserRole", "RolePermission",
    "StateTransition", "ExceptionQueue",
    "RecruitmentRequest", "PositionCompetency",
    "ResumeScore",
    "Interview", "InterviewerScore", "AIInterviewReport",
    "OfferApproval",
    "OfferTemplate", "OfferApprovalFlow", "OfferApprovalRecord", "OfferAttachment",
    "PositionAIArtifact",
    "ProbationPlan", "ProbationWeekReview", "ConfirmationReview",
    "TrainingCourse", "EmployeeTrainingProgress", "MentorRecord",
    "WorkTask", "TaskAcceptance", "PointRecord", "RewardPenaltyRecord", "Appeal",
    "TalentProfile", "AbilityTag", "PromotionRecord", "EquityRecord",
    "Project", "ProjectAssignment",
]
