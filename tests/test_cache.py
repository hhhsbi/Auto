"""
test_cache.py —— 两级缓存单元测试

测试 LRU 一级缓存读写、key 设计（权限/模型版本进 key）、
invalidate_doc 失效、clear_all 全清、stats 统计。

不依赖 Redis（CacheManager 自动降级为纯 LRU）。
"""
import json

import pytest

from cache import CacheManager, _sha1, _bucket, EMBEDDING_MODEL_VERSION


@pytest.fixture
def cache():
    """每个测试用全新的 CacheManager（Redis 不可用时自动降级 LRU）。
    Redis 可用时清掉 rag:*/ask:*/chat:* 残留数据，保证测试隔离。"""
    cm = CacheManager()
    # 清 Redis 残留（clear_all 不含 chat:*，这里补上 chat:* 并复用 clear_all）
    cm.clear_all()
    if cm._redis_ok and cm._redis is not None:
        try:
            for k in cm._redis.scan_iter(match="chat:*", count=500):
                cm._redis.delete(k)
        except Exception:
            pass
    # 清类级别共享的 _local_chat（降级模式下多实例共享同一 dict）
    CacheManager._local_chat.clear()
    return cm


class TestKeyDesign:
    """key 设计：权限/模型版本进 key，角色/部门不同自然 miss。"""

    def test_retrieve_key_contains_model_version(self, cache):
        key = cache._retrieve_key(_sha1("test"), _bucket("admin"), 5)
        assert EMBEDDING_MODEL_VERSION in key

    def test_retrieve_key_role_different(self, cache):
        """不同角色 key 不同（RBAC 权限隔离）。"""
        k_admin = cache._retrieve_key(_sha1("q"), _bucket("admin"), 5)
        k_hr = cache._retrieve_key(_sha1("q"), _bucket("hr"), 5)
        assert k_admin != k_hr

    def test_bucket_none_to_anon(self):
        assert _bucket(None) == "anon"

    def test_bucket_case_insensitive(self):
        assert _bucket("Admin") == _bucket("admin")

    def test_sha1_deterministic(self):
        assert _sha1("hello") == _sha1("hello")
        assert _sha1("hello") != _sha1("world")


class TestRetrieveCache:
    """retrieve 缓存读写 + 命中。"""

    def test_set_then_get_hits_lru(self, cache):
        docs = [{"text": "content1", "metadata": {"doc_id": "doc1"}, "distance": 0.1}]
        cache.set_retrieve("机器学习", 5, "admin", "tech", docs)
        result, source = cache.get_retrieve("机器学习", 5, "admin", "tech")
        assert result is not None
        assert result == docs
        assert source == "lru"

    def test_miss_returns_none(self, cache):
        result, source = cache.get_retrieve("不存在的问题", 5, "admin", "tech")
        assert result is None
        assert source is None

    def test_different_role_misses(self, cache):
        """admin 缓存的数据，hr 查不到（权限隔离）。"""
        docs = [{"text": "机密", "metadata": {"doc_id": "d1"}}]
        cache.set_retrieve("问题", 5, "admin", "tech", docs)
        result, _ = cache.get_retrieve("问题", 5, "hr", "tech")
        assert result is None

    def test_different_top_k_misses(self, cache):
        """top_k=5 的缓存，top_k=10 查不到。"""
        docs = [{"text": "x", "metadata": {"doc_id": "d1"}}]
        cache.set_retrieve("问题", 5, "admin", "tech", docs)
        result, _ = cache.get_retrieve("问题", 10, "admin", "tech")
        assert result is None


class TestAskCache:
    """/ask 缓存读写。"""

    def test_set_then_get_ask(self, cache):
        state = {"final_report": "报告内容", "iteration": 1}
        cache.set_ask("什么是AI", "admin", "tech", state)
        result, source = cache.get_ask("什么是AI", "admin", "tech")
        assert result == state
        assert source == "lru"

    def test_ask_department_isolation(self, cache):
        """不同部门 key 不同。"""
        cache.set_ask("问题", "admin", "tech", {"r": 1})
        result, _ = cache.get_ask("问题", "admin", "hr")
        assert result is None


class TestInvalidation:
    """缓存失效策略。"""

    def test_invalidate_doc_clears_related(self, cache):
        """set_retrieve 时建反向索引，invalidate_doc 后相关 key 被清。"""
        docs = [{"text": "x", "metadata": {"doc_id": "target_doc"}}]
        cache.set_retrieve("问题A", 5, "admin", "tech", docs)
        # 确认缓存命中
        assert cache.get_retrieve("问题A", 5, "admin", "tech")[0] is not None
        # 失效
        deleted = cache.invalidate_doc("target_doc")
        # 确认被清了
        result, _ = cache.get_retrieve("问题A", 5, "admin", "tech")
        assert result is None

    def test_clear_all(self, cache):
        """全清后 LRU 为空。"""
        cache.set_retrieve("问题", 5, "admin", "tech", [{"text": "x"}])
        cache.set_ask("问题2", "admin", "tech", {"r": 1})
        stats = cache.clear_all()
        assert stats["lru_keys"] >= 2
        assert cache.stats()["lru_size"] == 0


class TestStats:
    """stats() 返回结构。"""

    def test_stats_structure(self, cache):
        s = cache.stats()
        assert "model_version" in s
        assert "ttl_seconds" in s
        assert "lru_maxsize" in s
        assert "lru_size" in s
        assert "redis_enabled" in s
        assert s["model_version"] == EMBEDDING_MODEL_VERSION

    def test_stats_counts_after_set(self, cache):
        cache.set_retrieve("问题", 5, "admin", "tech", [{"text": "x"}])
        cache.set_ask("问题2", "admin", "tech", {"r": 1})
        s = cache.stats()
        assert s["lru_size"] == 2
        assert s["lru_retrieve_keys"] == 1
        assert s["lru_ask_keys"] == 1


class TestChatHistory:
    """多轮对话历史存储（Step 5）。"""

    def test_append_and_get_chat(self, cache):
        cache.append_chat("session1", "user", "你好")
        cache.append_chat("session1", "assistant", "你好呀！")
        history = cache.get_chat("session1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_get_empty_session(self, cache):
        assert cache.get_chat("不存在的session") == []

    def test_clear_chat(self, cache):
        cache.append_chat("s1", "user", "测试")
        cache.clear_chat("s1")
        assert cache.get_chat("s1") == []
