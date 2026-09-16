"""
cache.py —— 两级缓存 + 失效策略（阶段 4）

设计：
  一级（进程内）：cachetools.TTLCache，TTL 5 分钟兜底防脏数据
  二级（Redis）：跨进程共享，未启 / 异常时优雅降级到只用一级

Key 设计（权限 / 模型版本 / top_k 全进 key，自然失效）：
  retrieve: rag:emb:{query_sha1}:{model_version}:{role_bucket}:{top_k}
  /ask    : ask:{question_sha1}:{model_version}:{role_bucket}:{dept_bucket}
权限/模型变了 key 自然不同；role=None 用 "anon" 占位，未认证只命中公开检索的缓存。

失效策略：
  · 文件更新 → invalidate_doc(doc_id)：按 doc_id 反向索引找出命中过该文档的
    query_sha1 列表，再 SCAN 模糊删该 sha1 下所有 role/top_k 组合的 key
  · 权限变更 → role 在 key 里，自然失效
  · 模型/embedding 换版本 → EMBEDDING_MODEL_VERSION 常量改了全量失效
  · 手动清 → clear_all()：清 LRU + SCAN rag:* / ask:* 删 Redis
  · TTL 300s 兜底防漏失效的脏数据

反向索引：cache.set_retrieve 时记录 doc_id → set(query_sha1)（存 Redis SET，
LRU 端不维护反向索引——Redis 不可用时反向索引丢失，失效只能暴力清 LRU）。
"""
import hashlib
import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from cachetools import TTLCache

try:
    import redis as redis_lib
    HAS_REDIS_LIB = True
except ImportError:
    HAS_REDIS_LIB = False


# ────────────────────────── 全局常量 ──────────────────────────

# embedding 模型版本（rag.py 里改成同款常量；改模型就改这里，全部 key 自动失效）
EMBEDDING_MODEL_VERSION = "paraphrase-multilingual-MiniLM-L12-v2"

TTL_SECONDS = 300          # 5 分钟兜底
LRU_MAXSIZE = 256          # 一级缓存最多缓存多少个 key（retrieve + ask 合用）

# 从 .env 读 Redis 配置；不配就用默认值。生产可设 REDIS_URL=redis://user:pwd@host:6379/0
import os
from dotenv import load_dotenv
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None


# ────────────────────────── 工具函数 ──────────────────────────

def _sha1(text: str) -> str:
    """统一用 sha1（query/question 都较短，sha1 够安全且短）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _bucket(v: Optional[str]) -> str:
    """role / department 统一映射成 key 段：None → anon（未认证只命中公开缓存）。"""
    return (v or "anon").lower()


# ────────────────────────── CacheManager ──────────────────────────

class CacheManager:
    """两级缓存管理器（线程安全：LRU 加锁；Redis 自带连接池线程安全）。"""

    def __init__(self):
        # 一级：进程内 TTLCache；同一进程多线程复用
        self._lru: TTLCache = TTLCache(maxsize=LRU_MAXSIZE, ttl=TTL_SECONDS)
        self._lock = threading.Lock()
        # 二级：Redis（降级）
        self._redis = None
        self._redis_ok = False
        if HAS_REDIS_LIB:
            self._init_redis()

    def _init_redis(self):
        """连 Redis；连不上或不可用就降级（只一级缓存）。"""
        try:
            # protocol=2：兼容 Redis 5.x（不支持 RESP3 HELLO 握手）
            client = redis_lib.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                password=REDIS_PASSWORD,
                socket_connect_timeout=1, socket_timeout=1,
                protocol=2,
            )
            client.ping()
            self._redis = client
            self._redis_ok = True
            print(f"[cache] Redis 二级缓存已启用 {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
        except Exception as e:
            print(f"[cache] Redis 不可用，降级为只用一级 LRU：{e}")
            self._redis = None
            self._redis_ok = False

    # ───────── retrieve 缓存 ─────────

    def _retrieve_key(self, query_sha1: str, role_bucket: str, top_k: int) -> str:
        return f"rag:emb:{query_sha1}:{EMBEDDING_MODEL_VERSION}:{role_bucket}:{top_k}"

    def get_retrieve(self, query: str, top_k: int,
                     role: Optional[str], department: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """读 retrieve 缓存；命中返回结果，miss 返回 None。"""
        key = self._retrieve_key(_sha1(query), _bucket(role), top_k)
        # 一级
        with self._lock:
            v = self._lru.get(key)
        if v is not None:
            return v, "lru"  # (value, source) — source 用于验收看是否命中
        # 二级
        if self._redis_ok:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    val = json.loads(raw)
                    # 写回一级加速下次
                    with self._lock:
                        self._lru[key] = val
                    return val, "redis"
            except Exception:
                pass  # Redis 读失败不能影响主流程
        return None, None

    def set_retrieve(self, query: str, top_k: int,
                     role: Optional[str], department: Optional[str],
                     retrieved: List[Dict[str, Any]]):
        """写 retrieve 缓存 + 维护 doc_id 反向索引。"""
        key = self._retrieve_key(_sha1(query), _bucket(role), top_k)
        with self._lock:
            self._lru[key] = retrieved
        if not self._redis_ok:
            return
        # 二级 + 反向索引
        try:
            pipe = self._redis.pipeline()
            pipe.set(key, json.dumps(retrieved, ensure_ascii=False), ex=TTL_SECONDS)
            query_sha1 = _sha1(query)
            for res in retrieved:
                did = (res.get("metadata") or {}).get("doc_id")
                if did:
                    pipe.sadd(f"rag:doc:{did}", query_sha1)
                    pipe.expire(f"rag:doc:{did}", TTL_SECONDS * 2)  # 反向索引 TTL 略长
            pipe.execute()
        except Exception:
            pass  # Redis 写失败不影响主流程

    # ───────── /ask 缓存 ─────────

    def _ask_key(self, question_sha1: str, role_bucket: str, dept_bucket: str) -> str:
        return f"ask:{question_sha1}:{EMBEDDING_MODEL_VERSION}:{role_bucket}:{dept_bucket}"

    def get_ask(self, question: str, role: Optional[str],
                department: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        key = self._ask_key(_sha1(question), _bucket(role), _bucket(department))
        with self._lock:
            v = self._lru.get(key)
        if v is not None:
            return v, "lru"
        if self._redis_ok:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    val = json.loads(raw)
                    with self._lock:
                        self._lru[key] = val
                    return val, "redis"
            except Exception:
                pass
        return None, None

    def set_ask(self, question: str, role: Optional[str],
                department: Optional[str], final_state: Dict[str, Any]):
        key = self._ask_key(_sha1(question), _bucket(role), _bucket(department))
        with self._lock:
            self._lru[key] = final_state
        if not self._redis_ok:
            return
        try:
            self._redis.set(key, json.dumps(final_state, ensure_ascii=False, default=str),
                            ex=TTL_SECONDS)
        except Exception:
            pass

    # ───────── 失效 ─────────

    def invalidate_doc(self, doc_id: str) -> Dict[str, int]:
        """
        按 doc_id 失效：找反向索引里命中过该 doc 的 query_sha1 列表，
        删这些 sha1 下所有 role/top_k 组合的 key（rag:emb:{sha}:*）。
        Redis 不可用时（反向索引丢失）只能暴力清一级 retrieve 缓存。
        """
        deleted = {"lru_keys": 0, "redis_keys": 0}
        # 收集需要清的 query_sha1 集合
        shas: List[str] = []
        if self._redis_ok:
            try:
                members = self._redis.smembers(f"rag:doc:{doc_id}")
                shas = [m.decode() if isinstance(m, bytes) else m for m in members]
                # 模糊删 Redis：rag:emb:{sha}:*
                for sha in shas:
                    pattern = f"rag:emb:{sha}:*"
                    for k in self._redis.scan_iter(match=pattern, count=200):
                        self._redis.delete(k)
                        deleted["redis_keys"] += 1
                # 删反向索引本身
                self._redis.delete(f"rag:doc:{doc_id}")
            except Exception as e:
                print(f"[cache] invalidate_doc Redis 操作失败：{e}")
        # 一级：按 sha 前缀删 LRU
        with self._lock:
            for sha in shas:
                prefix = f"rag:emb:{sha}:"
                for k in list(self._lru.keys()):
                    if k.startswith(prefix):
                        del self._lru[k]
                        deleted["lru_keys"] += 1
            # Redis 不可用时反向索引丢失，只能暴力清 retrieve 缓存
            if not self._redis_ok:
                for k in list(self._lru.keys()):
                    if k.startswith("rag:emb:"):
                        del self._lru[k]
                        deleted["lru_keys"] += 1
        return deleted

    def clear_all(self) -> Dict[str, int]:
        """手动清缓存（DELETE /cache/clear）：清一级全部 + 二级 rag:*/ask:* 前缀。"""
        stats = {"lru_keys": 0, "redis_keys": 0}
        with self._lock:
            stats["lru_keys"] = len(self._lru)
            self._lru.clear()
        if self._redis_ok:
            try:
                for pattern in ("rag:emb:*", "rag:doc:*", "ask:*"):
                    for k in self._redis.scan_iter(match=pattern, count=500):
                        self._redis.delete(k)
                        stats["redis_keys"] += 1
            except Exception as e:
                print(f"[cache] clear_all Redis 操作失败：{e}")
        return stats

    # ───────── 状态查询 ─────────

    def stats(self) -> Dict[str, Any]:
        """供 GET /cache/stats 返回，方便验收看缓存命中/规模。"""
        with self._lock:
            lru_size = len(self._lru)
            lru_retrieve = sum(1 for k in self._lru if k.startswith("rag:emb:"))
            lru_ask = sum(1 for k in self._lru if k.startswith("ask:"))
        return {
            "model_version": EMBEDDING_MODEL_VERSION,
            "ttl_seconds": TTL_SECONDS,
            "lru_maxsize": LRU_MAXSIZE,
            "lru_size": lru_size,
            "lru_retrieve_keys": lru_retrieve,
            "lru_ask_keys": lru_ask,
            "redis_enabled": self._redis_ok,
            "redis_host": f"{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}" if self._redis_ok else None,
        }

    # ────────────────────────── 阶段5 Step 5：多轮对话上下文 ──────────────────────────
    # 存 chat:{session_id} → Redis List；无 Redis 时降级到进程内 dict（多 worker 不共享）
    _local_chat: dict = {}  # fallback：{session_id: [{role,content,ts}, ...]}

    def append_chat(self, session_id: str, role: str, content: str) -> None:
        """追加一条对话记录。role="user"/"assistant"；content 是消息文本。
        失败不上抛——会话历史缺失不影响主流程（最坏单轮退化）。"""
        from datetime import datetime
        item = {"role": role, "content": content, "ts": datetime.now().isoformat(timespec="seconds")}
        key = f"chat:{session_id}"
        try:
            if self._redis_ok and self._redis is not None:
                self._redis.rpush(key, json.dumps(item, ensure_ascii=False))
                # 会话历史最多保留 20 轮（40 条），防止无限增长
                self._redis.ltrim(key, -40, -1)
            else:
                self._local_chat.setdefault(session_id, []).append(item)
                # 同样最多 40 条
                if len(self._local_chat[session_id]) > 40:
                    self._local_chat[session_id] = self._local_chat[session_id][-40:]
        except Exception as e:
            print(f"[cache] append_chat 失败（session={session_id}）：{e}", file=sys.stderr)

    def get_chat(self, session_id: str) -> list:
        """取对话历史。返回 list[{role,content,ts}]；无记录返 []。"""
        key = f"chat:{session_id}"
        try:
            if self._redis_ok and self._redis is not None:
                raw = self._redis.lrange(key, 0, -1)
                return [json.loads(x) for x in raw]
            return self._local_chat.get(session_id, [])
        except Exception as e:
            print(f"[cache] get_chat 失败（session={session_id}）：{e}", file=sys.stderr)
            return []

    def clear_chat(self, session_id: str) -> bool:
        """清空指定会话历史。返回 True=有清到数据。"""
        key = f"chat:{session_id}"
        try:
            if self._redis_ok and self._redis is not None:
                n = self._redis.delete(key)
                return n > 0
            if session_id in self._local_chat:
                del self._local_chat[session_id]
                return True
            return False
        except Exception as e:
            print(f"[cache] clear_chat 失败（session={session_id}）：{e}", file=sys.stderr)
            return False


# ────────────────────────── 模块级单例 ──────────────────────────

cache = CacheManager()
