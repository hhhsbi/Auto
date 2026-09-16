"""
ingest.py —— 文件异步入库管道（阶段 1）

职责：
  1) SQLite 元信息存储：documents（文件元数据）+ tasks（处理任务与进度）
  2) process_upload：后台任务，读文件 → 切块 → 向量化 → 分批入库，
     每个阶段实时更新任务进度，供 GET /tasks/{task_id} 查询

设计说明：
  - doc_id 用文件内容 sha256：同一文件重传天然幂等（先删旧块再入库），
    也为阶段 4 的"缓存与失效"打好基础（按 doc_id 失效）。
  - SQLite 用短连接（每次操作开新连接），读写分别发生在请求线程池和
    后台任务线程，短事务不会互相长时间持锁。
"""
import hashlib
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import rag
# 阶段 4：文件重传/更新后按 doc_id 失效掉命中过该文档的检索缓存
from cache import cache

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ingest.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,           -- 内容 sha256，同时是向量库元数据里的 doc_id
    filename    TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'processing',  -- processing / completed / failed
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id    TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',        -- pending / processing / completed / failed
    stage      TEXT NOT NULL DEFAULT '排队中',          -- 当前处理阶段的可读描述
    progress   REAL NOT NULL DEFAULT 0,                -- 0-100
    error      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
-- 阶段5 Step 1：用户表（替代 auth.py 里写死的 _USERS 字典），密码用 bcrypt 哈希
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,                      -- bcrypt 哈希（60 字符）
    role          TEXT NOT NULL,
    department    TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);
-- 阶段5 Step 1：审计日志表——谁问了什么、看了哪些文档、何时调的 LLM（合规 + 优化数据）
CREATE TABLE IF NOT EXISTS audit_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT,
    role          TEXT,
    action        TEXT NOT NULL,                       -- /login /ask /search /upload /tasks /cache/clear
    question      TEXT,                               -- 用户的 question 或 query
    seen_doc_ids  TEXT,                               -- JSON 数组：本次检索命中的 doc_id 列表
    iteration     INTEGER,
    review_score  REAL,
    cache_source  TEXT,
    ip            TEXT,
    user_agent    TEXT,
    latency_ms    INTEGER,
    status_code   INTEGER,
    error         TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_username_time ON audit_logs(username, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_action_time   ON audit_logs(action, created_at);
"""


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db():
    """建表（幂等），应用启动时调用一次。"""
    with _conn() as c:
        c.executescript(_SCHEMA)


# ────────────────────────── 用户表（阶段5 Step 1） ──────────────────────────

# 演示用户 seed 配置：表空时自动插这些用户。生产应换真实注册流程。
# 这里只存配置，密码哈希在 seed_users_if_empty 里现算，避免模块级依赖 bcrypt。
_SEED_USERS = [
    {"username": "admin", "password": "admin123", "role": "admin", "department": "admin"},
    {"username": "alice", "password": "alice123", "role": "hr", "department": "hr"},
    {"username": "bob", "password": "bob123", "role": "employee", "department": "tech"},
]


def seed_users_if_empty():
    """users 表为空时 seed 三个演示用户（bcrypt 哈希）。

    调用方：app.py startup。bcrypt 在这里才 import，避免 ingest 模块级
    被无 bcrypt 环境加载失败。
    """
    try:
        import bcrypt
    except ImportError:
        print("[ingest] bcrypt 未安装，跳过 seed_users（pip install bcrypt）")
        return
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if n > 0:
            return  # 已有用户，不覆盖
        now = _now()
        for u in _SEED_USERS:
            pwd_hash = bcrypt.hashpw(u["password"].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            c.execute(
                "INSERT INTO users (username, password_hash, role, department, is_active, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (u["username"], pwd_hash, u["role"], u["department"], now),
            )
    print(f"[ingest] 已 seed {len(_SEED_USERS)} 个演示用户（admin/alice/bob）")


def get_user_by_username(username: str):
    """查单个用户；返回 dict（含 username/password_hash/role/department/is_active）或 None。"""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute(
            "SELECT username, password_hash, role, department, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def list_users():
    """列全部用户（不含 password_hash）。"""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT username, role, department, is_active, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def create_user(username: str, password_hash: str, role: str, department: str) -> bool:
    """新建用户；username 冲突返回 False。"""
    now = _now()
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users (username, password_hash, role, department, is_active, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (username, password_hash, role, department, now),
            )
        return True
    except sqlite3.IntegrityError:
        return False  # username 已存在


def update_user(username: str, **fields):
    """改用户字段（password_hash / role / department / is_active）。用户不存在返回 False。"""
    if not fields:
        return False
    allowed = {"password_hash", "role", "department", "is_active"}
    sets = ", ".join(f"{k} = ?" for k in fields if k in allowed)
    vals = [v for k, v in fields.items() if k in allowed]
    if not sets:
        return False
    with _conn() as c:
        cur = c.execute(f"UPDATE users SET {sets} WHERE username = ?", (*vals, username))
        return cur.rowcount > 0


def create_document(doc_id: str, filename: str, size_bytes: int, sha256: str):
    now = _now()
    with _conn() as c:
        # INSERT OR REPLACE：同一文件（doc_id=内容sha256）重复上传时幂等覆盖旧记录，
        # 后台任务会先清旧块再重新入库
        c.execute(
            "INSERT OR REPLACE INTO documents (doc_id, filename, size_bytes, sha256, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'processing', ?, ?)",
            (doc_id, filename, size_bytes, sha256, now, now),
        )


def create_task(doc_id: str) -> str:
    task_id = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO tasks (task_id, doc_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (task_id, doc_id, now, now),
        )
    return task_id


def _update_task(task_id: str, **fields):
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE tasks SET {sets} WHERE task_id = ?", (*fields.values(), task_id))


def _update_document(doc_id: str, **fields):
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE documents SET {sets} WHERE doc_id = ?", (*fields.values(), doc_id))


def get_task_detail(task_id: str) -> Optional[dict]:
    """查询任务详情，联表带出所属文档的元信息；任务不存在返回 None。"""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute(
            "SELECT t.task_id, t.doc_id, t.status, t.stage, t.progress, t.error, "
            "       t.created_at, t.updated_at, "
            "       d.filename, d.size_bytes, d.chunk_count "
            "FROM tasks t LEFT JOIN documents d ON d.doc_id = t.doc_id "
            "WHERE t.task_id = ?",
            (task_id,),
        ).fetchone()
    return dict(row) if row else None


def _decode(raw: bytes) -> str:
    """解码上传文件：先试 UTF-8，再试 GBK（Windows 常见编码），兜底替换非法字节。"""
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def process_upload(task_id: str, doc_id: str, filepath: str, source: str,
                   allowed_roles: str = "", allowed_departments: str = ""):
    """
    后台异步入库管道：解析 → 切块 → 向量化分批入库。

    由 FastAPI BackgroundTasks 在响应返回后的线程池里调用，
    任何异常都捕获并落到任务/文档状态上，不会向外抛。

    :param allowed_roles: 可见角色（逗号分隔原始串，如 "hr,admin"；空=公开）
    :param allowed_departments: 可见部门（逗号分隔原始串；空=公开）
    """
    try:
        _update_task(task_id, status="processing", stage="解析文件", progress=5)
        text = _decode(Path(filepath).read_bytes())

        _update_task(task_id, stage="切块", progress=15)
        chunks = rag.chunk_text(text)
        if not chunks:
            raise ValueError("文件没有可入库的文本内容")

        # 重传/重试幂等：先清掉同 doc_id 的旧块再入库
        _update_task(task_id, stage="清理旧块", progress=20)
        rag.delete_doc(doc_id)
        # 阶段 4：同步失效掉命中过该 doc_id 的检索缓存（按反向索引精确删，
        # 不命中过则 no-op；新文件 doc_id 是新 sha256，反向索引里没有，自然 no-op）
        try:
            cache.invalidate_doc(doc_id)
        except Exception as e:
            # 缓存失效失败不能让入库整体失败——TTL 5 分钟兜底
            print(f"[ingest] 缓存失效失败（doc_id={doc_id[:8]}...）: {e}")

        def progress_cb(done: int, total: int):
            _update_task(
                task_id,
                stage=f"向量化入库 ({done}/{total} 块)",
                progress=20 + 75 * done / total,
            )

        indexed = rag.index_chunks(
            doc_id, source=source, chunks=chunks,
            progress_cb=progress_cb,
            allowed_roles=allowed_roles, allowed_departments=allowed_departments,
        )

        _update_task(task_id, status="completed", stage="完成", progress=100)
        _update_document(doc_id, status="completed", chunk_count=indexed, error=None)
    except Exception as e:
        _update_task(task_id, status="failed", stage="失败", error=str(e))
        _update_document(doc_id, status="failed", error=str(e))
