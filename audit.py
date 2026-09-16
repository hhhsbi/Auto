"""
audit.py —— 审计日志（阶段5 Step 1）

职责：
  1) log_audit(...)：把每次接口调用记到 ingest.db 的 audit_logs 表——
     谁在何时调了什么接口、问了什么、看了哪些文档、耗时多少、是否出错。
  2) query_logs(...)：给 GET /audit/logs 查询用，支持按 username/action 过滤。

设计：
  - 复用 ingest.py 的 SQLite 短连接模式（_conn），表 schema 在 ingest._SCHEMA 里建。
  - 写日志失败绝不上抛：审计不能影响业务主流程，失败只打 stderr。
  - seen_doc_ids 是 JSON 数组：从 final_state["search_results"] 收集 doc_id 集合。
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingest import DB_PATH, _now


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def log_audit(
    username: Optional[str],
    role: Optional[str],
    action: str,
    question: Optional[str] = None,
    seen_doc_ids: Optional[List[str]] = None,
    iteration: Optional[int] = None,
    review_score: Optional[float] = None,
    cache_source: Optional[str] = None,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    latency_ms: Optional[int] = None,
    status_code: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """
    写一条审计日志到 audit_logs 表。

    任何异常都吞掉只打 stderr——审计日志失败不能影响业务主流程。
    """
    try:
        seen_json = json.dumps(seen_doc_ids or [], ensure_ascii=False)
        with _conn() as c:
            c.execute(
                "INSERT INTO audit_logs (username, role, action, question, seen_doc_ids, "
                "iteration, review_score, cache_source, ip, user_agent, latency_ms, "
                "status_code, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    username, role, action, question, seen_json,
                    iteration, review_score, cache_source, ip, user_agent,
                    latency_ms, status_code, error, _now(),
                ),
            )
    except Exception as e:
        print(f"[audit] 写日志失败：{e}", file=sys.stderr)


def query_logs(
    username: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    查审计日志（给 GET /audit/logs 用）。

    :param username: 按用户过滤（前缀匹配，None=不过滤）
    :param action: 按接口过滤（如 "/ask"，None=不过滤）
    :param limit: 最多返回条数（默认 100，最大 1000）
    """
    limit = max(1, min(limit, 1000))
    sql = "SELECT id, username, role, action, question, seen_doc_ids, iteration, "
    sql += "review_score, cache_source, ip, user_agent, latency_ms, status_code, error, created_at "
    sql += "FROM audit_logs"
    where = []
    args: List[Any] = []
    if username:
        where.append("username = ?")
        args.append(username)
    if action:
        where.append("action = ?")
        args.append(action)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(sql, args).fetchall()
    return [dict(r) for r in rows]
