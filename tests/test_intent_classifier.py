"""Intent classifier sanity tests."""

from app.agent.intent_classifier import classify_intent, get_tool_names_for_intent


def test_count_candidates_routes_to_dashboard():
    assert classify_intent("现在有多少份候选人") == "dashboard"
    tools = get_tool_names_for_intent("dashboard")
    assert "get_operations_dashboard" in tools
    assert "rag_search" not in tools


def test_status_change_routes_to_candidate_action():
    assert classify_intent("标记候选人一面未通过") == "candidate_action"
    tools = get_tool_names_for_intent("candidate_action")
    assert "update_resume" in tools
    assert "rag_search" not in tools
    assert "generate_questions" not in tools


def test_position_jd_routes_to_position_query():
    assert classify_intent("查看数据开发工程师岗位JD") == "position_query"
    tools = get_tool_names_for_intent("position_query")
    assert "get_position" in tools
    assert "list_resumes" not in tools


def test_resume_detail_routes_to_candidate_query():
    assert classify_intent("查看杨朝阳简历详情") == "candidate_query"
    tools = get_tool_names_for_intent("candidate_query")
    assert "get_resume" in tools


def test_position_other_candidates_routes_to_candidate_query():
    msg = "看看Agent工程师岗位的其他候选人"
    assert classify_intent(msg) == "candidate_query"
    tools = get_tool_names_for_intent("candidate_query")
    assert "list_resumes" in tools
    assert "get_questions" not in tools
    assert "generate_questions" not in tools


def test_resume_detail_excludes_position_questions():
    assert classify_intent("查看吴佳熙完整简历") == "candidate_query"
    tools = get_tool_names_for_intent("candidate_query")
    assert "get_resume" in tools
    assert "get_position_questions" not in tools


def test_interview_evaluation_routes_to_interview():
    assert classify_intent("开始面试评定") == "interview"
    tools = get_tool_names_for_intent("interview")
    assert "get_evaluation" in tools
