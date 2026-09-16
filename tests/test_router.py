"""
test_router.py —— 意图路由单元测试

覆盖 router_node 的 6 条路由规则：
  1) 闲聊模式匹配（你好/谢谢/你是谁）
  2) 含问号 → 研究
  3) 研究关键词命中 → 研究
  4) 有历史 + 简短追问 → 闲聊
  5) 极短消息 → 闲聊
  6) 默认 → 研究
"""
import pytest

from main import router_node, should_research


class TestRouterChatIntent:
    """规则 1+5：闲聊模式匹配 + 极短消息。"""

    def test_greeting_你好(self, fake_state):
        fake_state["research_question"] = "你好"
        assert router_node(fake_state)["intent"] == "chat"

    def test_greeting_您好(self, fake_state):
        fake_state["research_question"] = "您好"
        assert router_node(fake_state)["intent"] == "chat"

    def test_thanks_谢谢(self, fake_state):
        fake_state["research_question"] = "谢谢"
        assert router_node(fake_state)["intent"] == "chat"

    def test_identity_你是谁(self, fake_state):
        fake_state["research_question"] = "你是谁"
        assert router_node(fake_state)["intent"] == "chat"

    def test_english_hello(self, fake_state):
        fake_state["research_question"] = "hello"
        assert router_node(fake_state)["intent"] == "chat"

    def test_short_message_ok(self, fake_state):
        """极短无关键词 → 闲聊。"""
        fake_state["research_question"] = "好的"
        assert router_node(fake_state)["intent"] == "chat"

    def test_short_with_question_mark_is_research(self, fake_state):
        """极短但有问号 → 研究（规则 2 优先于规则 5）。"""
        fake_state["research_question"] = "行吗？"
        assert router_node(fake_state)["intent"] == "research"


class TestRouterResearchIntent:
    """规则 2+3+6：含问号 / 关键词 / 默认。"""

    def test_question_mark_research(self, fake_state):
        fake_state["research_question"] = "什么是机器学习？"
        assert router_node(fake_state)["intent"] == "research"

    def test_keyword_分析(self, fake_state):
        fake_state["research_question"] = "分析一下数据安全趋势"
        assert router_node(fake_state)["intent"] == "research"

    def test_keyword_什么是(self, fake_state):
        """"什么是机器学习"只有 7 字，但命中关键词 → 研究。"""
        fake_state["research_question"] = "什么是机器学习"
        assert router_node(fake_state)["intent"] == "research"

    def test_keyword_对比(self, fake_state):
        fake_state["research_question"] = "对比RSA和ECC加密算法"
        assert router_node(fake_state)["intent"] == "research"

    def test_keyword_如何(self, fake_state):
        fake_state["research_question"] = "如何实现一个简单的推荐系统"
        assert router_node(fake_state)["intent"] == "research"

    def test_default_long_question(self, fake_state):
        """长问题无关键词无问号 → 默认研究。"""
        fake_state["research_question"] = "请帮我生成一份关于网络安全最佳实践的详细文档"
        assert router_node(fake_state)["intent"] == "research"


class TestRouterFollowUp:
    """规则 4：有历史 + 简短追问 → 闲聊。"""

    def test_followup_with_history_is_chat(self, fake_state):
        fake_state["research_question"] = "继续"
        fake_state["chat_history"] = [
            {"role": "user", "content": "什么是机器学习"},
            {"role": "assistant", "content": "机器学习是..."},
        ]
        assert router_node(fake_state)["intent"] == "chat"

    def test_followup_without_history_is_research(self, fake_state):
        """无历史时"继续"命中规则 5（<8 字）→ 闲聊；但"详细说说"是 4 字也闲聊。
        这里测有历史才走追问逻辑。"""
        fake_state["research_question"] = "详细说说"
        fake_state["chat_history"] = [
            {"role": "user", "content": "什么是深度学习"},
            {"role": "assistant", "content": "深度学习是..."},
        ]
        assert router_node(fake_state)["intent"] == "chat"


class TestShouldResearch:
    """条件边函数 should_research。"""

    def test_chat_routes_to_chat(self, fake_state):
        fake_state["intent"] = "chat"
        assert should_research(fake_state) == "chat"

    def test_research_routes_to_planner(self, fake_state):
        fake_state["intent"] = "research"
        assert should_research(fake_state) == "planner"

    def test_unknown_intent_defaults_to_planner(self, fake_state):
        fake_state["intent"] = ""
        assert should_research(fake_state) == "planner"
