"""Agent Tool Registry — all tools the AI Agent can call."""
import json
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import async_session_factory


async def _get_db() -> AsyncSession:
    async with async_session_factory() as session:
        return session


# ── Tool Definitions ────────────────────────────────────

TOOL_REGISTRY = {
    # Recruitment tools
    "list_resumes": {
        "name": "list_resumes",
        "description": "查询候选人列表。可按岗位、状态、关键词筛选，支持排序和分页。",
        "parameters": {
            "type": "object",
            "properties": {
                "positionId": {"type": "string", "description": "岗位ID，不传表示全部"},
                "statuses": {"type": "string", "description": "状态列表，逗号分隔"},
                "keyword": {"type": "string", "description": "搜索关键词"},
                "sortBy": {"type": "string", "enum": ["score", "uploadTime"], "description": "排序字段"},
                "limit": {"type": "integer", "description": "返回数量限制，默认10"},
            },
        },
    },
    "get_resume": {
        "name": "get_resume",
        "description": "获取单个候选人的详细信息，包括AI分析结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人ID"},
            },
            "required": ["id"],
        },
    },
    "list_positions": {
        "name": "list_positions",
        "description": "获取所有岗位列表。",
        "parameters": {"type": "object", "properties": {}},
    },
    "update_resume": {
        "name": "update_resume",
        "description": "更新候选人信息或状态。可更新基本信息、技能、工作经历等。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人ID"},
                "status": {"type": "string", "description": "新状态：job_hunting/passed/first_interview/..."},
                "fields": {"type": "object", "description": "要更新的字段"},
            },
            "required": ["id"],
        },
    },

    # Interview tools
    "get_questions": {
        "name": "get_questions",
        "description": "获取或自动生成面试题单。若题单不存在会自动调用AI生成。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {"type": "string", "enum": ["first", "second"]},
            },
            "required": ["candidateId", "round"],
        },
    },
    "generate_questions": {
        "name": "generate_questions",
        "description": "为候选人重新生成面试题目。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {"type": "string", "enum": ["first", "second"]},
            },
            "required": ["candidateId", "round"],
        },
    },
    "get_evaluation": {
        "name": "get_evaluation",
        "description": "获取候选人的面试评分详情。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {"type": "string"},
            },
            "required": ["candidateId"],
        },
    },
    "ai_score_question": {
        "name": "ai_score_question",
        "description": "使用AI对面试回答进行评分。",
        "parameters": {
            "type": "object",
            "properties": {
                "questionId": {"type": "string"},
                "answer": {"type": "string", "description": "候选人的回答"},
            },
            "required": ["questionId"],
        },
    },
    "get_leaderboard": {
        "name": "get_leaderboard",
        "description": "获取一面/二面排行榜。",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["first_result", "second_result"]},
            },
            "required": ["category"],
        },
    },

    # Probation/Performance
    "list_probation": {
        "name": "list_probation",
        "description": "查询试用期员工列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "department": {"type": "string"},
                "status": {"type": "string"},
            },
        },
    },
    "list_performance": {
        "name": "list_performance",
        "description": "查询绩效数据。",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string", "description": "如：2026-Q3"},
            },
            "required": ["quarter"],
        },
    },

    # Knowledge / RAG
    "rag_search": {
        "name": "rag_search",
        "description": "在知识库中检索相关内容（RAG）。用于查面试题、公司制度、技术规范等。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询文本"},
                "topK": {"type": "integer", "description": "返回条数，默认5"},
                "categoryKey": {"type": "string", "description": "分类过滤"},
            },
            "required": ["query"],
        },
    },
    "list_knowledge": {
        "name": "list_knowledge",
        "description": "查询知识库素材列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "categoryKey": {"type": "string"},
                "keyword": {"type": "string"},
            },
        },
    },

    # Dashboard
    "get_operations_dashboard": {
        "name": "get_operations_dashboard",
        "description": "获取运营看板数据。包含概览统计、风险、漏斗等。",
        "parameters": {"type": "object", "properties": {}},
    },

    # System
    "get_settings": {
        "name": "get_settings",
        "description": "获取系统设置。",
        "parameters": {"type": "object", "properties": {}},
    },
}


def get_tools_for_agent(agent_id: str = "recruit") -> List[dict]:
    """Get the list of tool definitions for a specific agent type."""
    recruit_tools = ["list_resumes", "get_resume", "list_positions", "update_resume", "rag_search"]
    interview_tools = ["get_questions", "generate_questions", "get_evaluation", "ai_score_question", "get_leaderboard", "rag_search"]
    training_tools = ["list_probation", "rag_search"]
    performance_tools = ["list_performance", "rag_search"]

    tool_map = {
        "recruit": recruit_tools,
        "interview": interview_tools,
        "training": training_tools,
        "performance": performance_tools,
    }

    # All agents get some common tools
    common = ["get_operations_dashboard", "get_settings", "list_knowledge"]
    names = tool_map.get(agent_id, recruit_tools) + common

    return [TOOL_REGISTRY[name] for name in names if name in TOOL_REGISTRY]


# ── LangChain / LangGraph Tool Conversion ────────────────

import inspect
from typing import get_type_hints
from pydantic import create_model, Field


_JSON_TYPE_MAP = {
    "string": (str, None),
    "integer": (int, None),
    "number": (float, None),
    "boolean": (bool, None),
    "object": (dict, None),
    "array": (list, None),
}


def _build_pydantic_model(tool_name: str, params_schema: dict) -> type:
    """Convert a JSON Schema parameters dict into a dynamic Pydantic model."""
    fields = {}
    properties = params_schema.get("properties", {})
    required = params_schema.get("required", [])

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        python_type, _ = _JSON_TYPE_MAP.get(json_type, (str, None))
        description = prop_schema.get("description", "")
        default = ... if prop_name in required else None
        fields[prop_name] = (python_type, Field(default, description=description))

    model_name = f"{tool_name}_params"
    # If no fields, create a model with no required fields (empty input)
    if not fields:
        fields["dummy"] = (str | None, Field(None, description="No parameters needed"))

    return create_model(model_name, **fields)


async def _execute_tool_sync(tool_name: str, **kwargs) -> str:
    """Execute a tool with an auto-created DB session (for LangGraph context).

    Uses lazy import to avoid circular dependency with app.routers.ai_agent.
    """
    from app.routers.ai_agent import execute_tool_call  # lazy import

    # Remove None values (unset optional params)
    params = {k: v for k, v in kwargs.items() if v is not None}
    # Remove the dummy field if present
    params.pop("dummy", None)

    async with async_session_factory() as db:
        return await execute_tool_call(tool_name, params, db)


def create_langchain_tools(agent_id: str = "recruit") -> list:
    """Convert the tool registry into LangChain StructuredTool objects.

    Returns a list of tools compatible with LangGraph's ToolNode.
    """
    from langchain_core.tools import StructuredTool

    tool_defs = get_tools_for_agent(agent_id)
    lc_tools = []

    for td in tool_defs:
        tool_name = td["name"]
        description = td["description"]
        params_schema = td.get("parameters", {})

        # Build Pydantic args model from JSON Schema
        args_model = _build_pydantic_model(tool_name, params_schema)

        # Create the async tool function bound to this tool_name
        async def tool_func(tool_name=tool_name, **kwargs) -> str:
            return await _execute_tool_sync(tool_name, **kwargs)

        # Attach proper signature for LangChain
        tool_func.__name__ = tool_name
        tool_func.__doc__ = description

        structured_tool = StructuredTool.from_function(
            name=tool_name,
            description=description,
            args_schema=args_model,
            coroutine=tool_func,
        )
        lc_tools.append(structured_tool)

    return lc_tools
