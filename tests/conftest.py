"""
conftest.py —— pytest 全局夹具

把重依赖（DeepSeek API、Chroma 向量库、SQLite 用户表）mock 掉，
让单测可以在没有 API Key、没有数据库的环境下跑。
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# 把项目根目录加到 sys.path，让 import main / import cache 能找到
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 测试环境：没有 API Key 也要能 import main（路由逻辑不调 LLM）
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


@pytest.fixture
def fake_state():
    """构造一个最小可用的 ResearchState，供 router 测试用。"""
    return {
        "research_question": "",
        "sub_questions": [],
        "search_results": [],
        "draft": None,
        "review_feedback": None,
        "iteration": 0,
        "final_report": None,
        "code_outputs": [],
        "code_images": [],
        "execution_log": "",
        "review_score": None,
        "user_role": None,
        "user_department": None,
        "chat_history": [],
        "session_id": None,
        "planner_iteration": 0,
        "intent": "",
    }
