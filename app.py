"""
app.py —— AutoResearch 的 FastAPI 服务层

  POST /login           —— 登录换 JWT（演示账号 admin/admin123、alice/alice123、bob/bob123）
  GET  /health          —— 健康检查（含知识库块数等运行状态，无需登录）
  POST /ask             —— 输入研究问题，同步返回最终报告（需登录；检索范围按角色/部门过滤）
  POST /search          —— 直接检索知识库（需登录；用于验证不同角色检索结果不同）
  POST /upload          —— 上传知识库文件（仅 admin；立即返回 task_id，后台异步切块入库）
  GET  /tasks/{task_id} —— 查询文件处理的阶段/进度/状态（需登录）

阶段5 Step 3+4 新增：
  GET  /audit/logs      —— 查审计日志（仅 admin）
  POST /users           —— 创建用户（仅 admin）
  GET  /users           —— 列出全部用户（仅 admin）
  PATCH /users/{username} —— 改密/改角色（仅 admin）

阶段5 Step 4 全局基础设施：
  - 结构化日志 → logs/app.log（RotatingFileHandler）
  - 全局异常中间件：未捕获异常 → 500 + 审计日志
  - slowapi 限流：/ask 5/min、/search 30/min、/upload 10/min（按用户名限）

启动：
  uvicorn app:app --reload

验证（先拿 token，再带着访问）：
  TOKEN=$(curl -s -X POST localhost:8000/login -H "Content-Type: application/json" \
       -d '{"username":"alice","password":"alice123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
  curl localhost:8000/health
  curl -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
       --data-binary @req.json
  curl -X POST localhost:8000/search -H "Authorization: Bearer $TOKEN" \
       -H "Content-Type: application/json" -d '{"query":"公司制度"}'
  curl -X POST "localhost:8000/upload?roles=hr" -H "Authorization: Bearer $ADMIN_TOKEN" \
       -F "file=@docs/宠物百科.txt"
  curl localhost:8000/tasks/<task_id> -H "Authorization: Bearer $TOKEN"
"""
import hashlib
import logging
import logging.handlers
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# import main 会在模块加载时完成：加载 DeepSeek key、加载 embedding 模型、
# 索引 docs/ 知识库（幂等，集合非空则跳过）。模型加载约需十几秒，属于启动成本。
from main import run_research, ENABLE_WEB_SEARCH, MAX_ITERATIONS  # noqa: F401
from main import run_research_stream, resume_research_stream  # 阶段5 Step 7：SSE 流式 + HITL
from rag import collection, retrieve
from ingest import (
    init_db, create_document, create_task, get_task_detail, process_upload,
    seed_users_if_empty, list_users, create_user, update_user,
)
from auth import (
    LoginRequest, UserContext, CreateUserRequest, UpdateUserRequest,
    authenticate, create_access_token, hash_password,
    get_current_user, require_admin,
)
# 阶段 4：/ask 结果缓存 + 失效管理（main 模块加载时已构造好 cache 单例）
from cache import cache as cache_mgr
# 阶段5 Step 3：审计日志
from audit import log_audit, query_logs

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

# 允许上传的知识库文件类型（与 rag.index_documents 支持的格式保持一致）
ALLOWED_EXTENSIONS = {".txt", ".md"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


# ────────────────────────── 阶段5 Step 4：结构化日志 ──────────────────────────

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
# RotatingFileHandler：单文件 10MB，最多保留 5 个备份
_file_handler = logging.handlers.RotatingFileHandler(
    LOGS_DIR / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
))
_logger = logging.getLogger("autoresearch")
_logger.setLevel(logging.INFO)
_logger.addHandler(_file_handler)
# 同时输出到控制台，方便本地开发看
_logger.addHandler(logging.StreamHandler())


def _log(msg: str, level: str = "info", **kwargs):
    """结构化日志：msg 是人话，kwargs 是结构化字段（user/action/latency 等）。"""
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    getattr(_logger, level)(f"{msg} | {extra}" if extra else msg)


# ────────────────────────── 阶段5 Step 4：slowapi 限流 ──────────────────────────

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False


def _user_key(request: Request) -> str:
    """限流 key：已登录用 username，未登录用 IP。

    必须在 _limiter 创建前定义——Limiter 的 key_func 在构造时传入。
    中间件层早于 FastAPI Depends，拿不到 user 对象，从 Authorization
    header 解 JWT 取 username。
    """
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        try:
            import jwt as _jwt
            from auth import JWT_SECRET, JWT_ALGORITHM
            payload = _jwt.decode(auth[7:], JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload.get("sub") or get_remote_address(request)
        except Exception:
            return get_remote_address(request)
    return get_remote_address(request)


# Limiter storage：配了 Redis 走 Redis（跨进程共享计数），否则用内存
_redis_url = os.getenv("REDIS_URL") or (
    f"redis://{os.getenv('REDIS_HOST', '127.0.0.1')}:{os.getenv('REDIS_PORT', '6379')}/1"
) if os.getenv("REDIS_HOST") else "memory://"

_limiter = None
if HAS_SLOWAPI:
    try:
        _limiter = Limiter(key_func=_user_key, storage_uri=_redis_url)
    except Exception as e:
        print(f"[app] slowapi 初始化失败，跳过限流：{e}")
        HAS_SLOWAPI = False


# ────────────────────────── Pydantic 请求 / 响应模型 ──────────────────────────

# ---------------------- Pydantic 请求 / 响应模型 ----------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="研究问题，如：猫有哪些生活习性？",
        examples=["猫有哪些生活习性？为什么说猫比较适合忙碌人士饲养？"],
    )


class AskResponse(BaseModel):
    question: str
    report: str  # 评审合格（或达到最大轮次）的最终报告，含 [来源N] 引用
    sub_questions: List[str]  # planner 拆解出的子问题
    iteration: int  # 反思循环轮数
    review_score: Optional[float] = None  # 评审打分（0-10）
    code_images: List[str] = []  # 代码执行生成的图表路径
    execution_log: str = ""  # 代码执行状态日志
    cache_source: Optional[str] = None  # 阶段4：本次响应来源 "lru"/"redis"/None(重新跑流水线)


class UploadResponse(BaseModel):
    task_id: str  # 异步处理任务 ID，用 GET /tasks/{task_id} 查进度
    doc_id: str  # 文档 ID（内容 sha256），知识库中该文件所有块的元数据标识
    filename: str
    size_bytes: int
    allowed_roles: str  # 入库时打的可见角色标签（逗号围栏串，空=公开）
    allowed_departments: str  # 入库时打的可见部门标签（空=公开）
    status: str = "pending"  # 初始状态：pending → processing → completed / failed
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    doc_id: str
    status: str  # pending / processing / completed / failed
    stage: str  # 当前阶段的可读描述
    progress: float  # 0-100
    filename: Optional[str] = None
    size_bytes: Optional[int] = None
    chunk_count: Optional[int] = None  # 处理完成后为入库块数
    error: Optional[str] = None
    created_at: str
    updated_at: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    department: str
    expires_note: str = "Token 默认 2 小时有效"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="检索问题")
    top_k: int = Field(5, ge=1, le=20, description="返回条数")


class SearchHit(BaseModel):
    source: str
    chunk_index: Optional[int] = None
    text: str  # 命中的块正文（截断到 300 字）
    distance: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    viewer: str  # 当前用户 "角色@部门"
    total: int
    hits: List[SearchHit]


class HealthResponse(BaseModel):
    status: str
    version: str
    knowledge_chunks: int  # 知识库已索引的文档块数
    web_search_enabled: bool
    max_iterations: int


# 阶段5 Step 3：审计日志查询响应模型
class AuditLogEntry(BaseModel):
    id: int
    username: Optional[str] = None
    role: Optional[str] = None
    action: str
    question: Optional[str] = None
    seen_doc_ids: Optional[str] = None  # JSON 数组字符串
    iteration: Optional[int] = None
    review_score: Optional[float] = None
    cache_source: Optional[str] = None
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    latency_ms: Optional[int] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    created_at: str


# 阶段5 Step 2：用户管理响应模型
class UserResponse(BaseModel):
    username: str
    role: str
    department: str
    is_active: int
    created_at: str


class CreateUserResponse(BaseModel):
    username: str
    role: str
    department: str
    message: str = "用户已创建"


class UpdateUserResponse(BaseModel):
    username: str
    updated: List[str]  # 实际更新的字段名列表
    message: str = "用户已更新"


# ---------------------- 应用与全局资源 ----------------------

# 初始化 SQLite 元信息表（幂等）；流水线图由 main.run_research 内部缓存复用
init_db()
# 阶段5 Step 2：seed 演示用户（users 表空时自动插 admin/alice/bob，bcrypt 哈希）
seed_users_if_empty()

app = FastAPI(
    title="AutoResearch RAG Service",
    description="多智能体深度研究系统：规划/检索/代码执行/写作/评审流水线的 HTTP 服务",
    version="0.2.0",  # 阶段5 提升 minor 版本
)

# 挂载前端静态文件（访问 http://127.0.0.1:8000/ 即打开聊天页面）
from fastapi.staticfiles import StaticFiles
from pathlib import Path as _Path
_static_dir = _Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/")
    def index_page():
        """前端聊天页面：登录 + 多轮对话 + SSE 流式 + HITL 介入。"""
        from fastapi.responses import FileResponse
        return FileResponse(str(_static_dir / "index.html"))


# ────────────────────────── 阶段5 Step 4：限流 & 全局异常中间件 ──────────────────────────

if HAS_SLOWAPI and _limiter is not None:
    # 自实现限流：用 Redis INCR + EXPIRE 滑动窗口计数（slowapi 内部 API 太隐晦，
    # 直接用 storage 层最稳）。无 Redis 时用进程内 dict。
    import time as _time
    _local_rate_counter: dict = {}  # fallback：{(path,key,minute): count}

    @app.middleware("http")
    async def _rate_limit_middleware(request: Request, call_next):
        """限流中间件：/ask 5/min、/search 30/min、/upload 10/min（按用户名限）。

        实现：每 (path, user, minute) 一个计数器，第 N 次请求 N > limit 时返 429。
        Redis 用 INCR + EXPIRE 60s；无 Redis 用进程内 dict（重启即清）。
        """
        path = request.url.path
        limits = {"ask": 5, "search": 30, "upload": 10}  # path 末段 → 上限/分钟
        seg = path.strip("/").split("/")[0]  # "/ask" → "ask"
        if seg not in limits:
            return await call_next(request)
        limit_n = limits[seg]
        user = _user_key(request)
        # minute 桶：当前 unix 时间除以 60，同一分钟内累加
        bucket = int(_time.time() // 60)
        rkey = f"ratelimit:{seg}:{user}:{bucket}"

        try:
            from cache import cache as _cache_mod  # 复用 cache 单例的 Redis 连接
            if _cache_mod._redis is not None:
                pipe = _cache_mod._redis.pipeline()
                pipe.incr(rkey)
                pipe.expire(rkey, 61)  # 略大于一分钟，确保桶过期前不会被误判
                count, _ = pipe.execute()
            else:
                raise RuntimeError("no redis")
        except Exception:
            # fallback：进程内 dict（多 worker 不共享，但单进程够用）
            count = _local_rate_counter.get(rkey, 0) + 1
            _local_rate_counter[rkey] = count
            # 简单清理：dict 超过 10000 项时清掉早于当前桶的
            if len(_local_rate_counter) > 10000:
                _local_rate_counter = {k: v for k, v in _local_rate_counter.items()
                                       if k.endswith(f":{bucket}")}

        if count > limit_n:
            _log("限流触发", level="warning", user=user, path=path,
                 count=count, limit=limit_n)
            return JSONResponse(
                status_code=429,
                content={"detail": f"请求过频：{seg} 上限 {limit_n}/分钟（用户 {user}，当前 {count}）"},
            )
        return await call_next(request)

    _log("限流已启用：/ask 5/min、/search 30/min、/upload 10/min（按用户名）")


@app.middleware("http")
async def _global_exception_middleware(request: Request, call_next):
    """全局异常兜底：未捕获异常 → 500 + 审计日志 + 结构化日志。"""
    t0 = time.time()
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # 从 Authorization header 解 username 尽力记录
        username = None
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            try:
                import jwt as _jwt
                from auth import JWT_SECRET, JWT_ALGORITHM
                payload = _jwt.decode(auth[7:], JWT_SECRET, algorithms=[JWT_ALGORITHM])
                username = payload.get("sub")
            except Exception:
                pass
        latency_ms = int((time.time() - t0) * 1000)
        err_msg = f"{type(e).__name__}: {e}"
        _log("未捕获异常", level="error", user=username, path=request.url.path,
             latency_ms=latency_ms, error=err_msg)
        log_audit(
            username=username, role=None, action=request.url.path,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            latency_ms=latency_ms, status_code=500, error=err_msg,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"服务器内部错误: {type(e).__name__}"},
        )


@app.on_event("startup")
async def _on_startup():
    """启动钩子：seed 用户（再次幂等）+ 打日志。"""
    seed_users_if_empty()
    _log("服务启动", action="startup", chunks=collection.count())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查：进程存活 + 知识库状态，不触发任何重计算，可接负载探针（无需鉴权）。"""
    return HealthResponse(
        status="ok",
        version=app.version,
        knowledge_chunks=collection.count(),
        web_search_enabled=ENABLE_WEB_SEARCH,
        max_iterations=MAX_ITERATIONS,
    )


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, request: Request) -> LoginResponse:
    """
    登录换取 JWT（无需鉴权）：

      curl -X POST localhost:8000/login -H "Content-Type: application/json" \
           -d '{"username":"alice","password":"alice123"}'

    演示账号：admin/admin123（全库可见，可上传）、alice/alice123（hr）、bob/bob123（tech 部门员工）。
    JWT 的 claims 带 role / department，后续请求放 Authorization: Bearer <token>。
    """
    t0 = time.time()
    user = authenticate(req.username, req.password)
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    latency_ms = int((time.time() - t0) * 1000)
    if user is None:
        # 失败也要审计（暴力破解检测）
        log_audit(username=req.username, role=None, action="/login",
                  question=None, ip=ip, user_agent=ua,
                  latency_ms=latency_ms, status_code=401, error="用户名或密码错误")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    log_audit(username=user.username, role=user.role, action="/login",
              ip=ip, user_agent=ua, latency_ms=latency_ms, status_code=200)
    _log("用户登录", user=user.username, role=user.role, latency_ms=latency_ms)
    return LoginResponse(
        access_token=create_access_token(user),
        username=user.username,
        role=user.role,
        department=user.department,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request,
        user: UserContext = Depends(get_current_user)) -> AskResponse:
    """
    同步执行完整研究流水线（拆解→检索→代码执行→写作→评审循环）。

    需要登录（Bearer Token）。检索范围按用户的角色/部门过滤：
    hr 用户看不到 tech 部门受限文档，反之亦然；admin 全库可见。
    耗时约 1~3 分钟，客户端需设置足够长的超时（如 curl --max-time 300）。
    全链路自动上报 LangSmith（run_research 为根 trace）。

    阶段 4 缓存：question+role+department+model_version 完全一致才命中，
    命中即复用上次流水线终态（响应 cache_source 标 "lru"/"redis"）；
    miss 才跑流水线，跑完写回两级缓存。

    阶段5 Step 3：审计日志记录 question/seen_doc_ids/iteration/review_score/cache_source。
    阶段5 Step 5：可选 X-Session-Id header 开多轮对话——同一 session_id 的问答会
    拼接历史到 writer prompt；不带 session_id 仍单轮（保持兼容）。
    """
    t0 = time.time()
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    # 阶段5 Step 5：取 session_id + 历史
    session_id = request.headers.get("x-session-id")
    history = cache_mgr.get_chat(session_id) if session_id else []

    # 1) 先查 /ask 缓存：命中直接返，跳过整个流水线
    # 注意：多轮对话不查缓存——同一 session 不同轮次 history 不同，缓存命中会破坏多轮语义
    cached_state, hit_source = (None, None)
    if not session_id:
        cached_state, hit_source = cache_mgr.get_ask(req.question, user.role, user.department)
    if cached_state:
        report = (cached_state.get("final_report")
                  or cached_state.get("draft") or "")
        if report.strip():
            # 从缓存终态抽 seen_doc_ids（用于审计——记录用户看了哪些文档）
            seen = sorted({
                (r.get("metadata") or {}).get("doc_id")
                for r in cached_state.get("search_results", [])
                if (r.get("metadata") or {}).get("doc_id")
            })
            log_audit(username=user.username, role=user.role, action="/ask",
                      question=req.question, seen_doc_ids=seen,
                      iteration=cached_state.get("iteration"),
                      review_score=cached_state.get("review_score"),
                      cache_source=hit_source,
                      ip=ip, user_agent=ua,
                      latency_ms=int((time.time()-t0)*1000),
                      status_code=200)
            return AskResponse(
                question=req.question,
                report=report,
                sub_questions=cached_state.get("sub_questions", []),
                iteration=cached_state.get("iteration", 0),
                review_score=cached_state.get("review_score"),
                code_images=cached_state.get("code_images", []),
                execution_log=cached_state.get("execution_log", ""),
                cache_source=hit_source,
            )
        # 缓存里 report 为空 → 视为脏数据，继续走流水线重算

    # 2) 缓存 miss → 跑完整流水线（含多轮对话历史）
    try:
        final_state = run_research(req.question, role=user.role, department=user.department,
                                   history=history, session_id=session_id)
    except Exception as e:
        # LLM API 故障、检索异常等统一转成 500，不把堆栈直接暴露给调用方
        log_audit(username=user.username, role=user.role, action="/ask",
                  question=req.question, ip=ip, user_agent=ua,
                  latency_ms=int((time.time()-t0)*1000),
                  status_code=500, error=f"流水线执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"流水线执行失败: {e}")

    # 评审合格会写 final_report；兜底取草稿，避免极端情况下返回空
    report = final_state.get("final_report") or final_state.get("draft") or ""
    if not report.strip():
        log_audit(username=user.username, role=user.role, action="/ask",
                  question=req.question, ip=ip, user_agent=ua,
                  latency_ms=int((time.time()-t0)*1000),
                  status_code=502, error="流水线结束但未生成有效报告")
        raise HTTPException(status_code=502, detail="流水线结束但未生成有效报告")

    # 3) 写回两级缓存（单轮才写——多轮历史不同，缓存会破坏语义）
    if not session_id:
        cache_mgr.set_ask(req.question, user.role, user.department, final_state)

    # 4) 阶段5 Step 5：多轮对话——把这次问答追加到 Redis 历史
    if session_id:
        cache_mgr.append_chat(session_id, "user", req.question)
        cache_mgr.append_chat(session_id, "assistant", report)

    # 5) 审计：从 search_results 抽 doc_id 集合
    seen = sorted({
        (r.get("metadata") or {}).get("doc_id")
        for r in final_state.get("search_results", [])
        if (r.get("metadata") or {}).get("doc_id")
    })
    latency_ms = int((time.time()-t0)*1000)
    log_audit(username=user.username, role=user.role, action="/ask",
              question=req.question, seen_doc_ids=seen,
              iteration=final_state.get("iteration"),
              review_score=final_state.get("review_score"),
              cache_source=None,
              ip=ip, user_agent=ua,
              latency_ms=latency_ms, status_code=200)
    _log("研究流水线完成", user=user.username, question_len=len(req.question),
         iteration=final_state.get("iteration"), seen_docs=len(seen),
         session_id=session_id, latency_ms=latency_ms)

    return AskResponse(
        question=req.question,
        report=report,
        sub_questions=final_state.get("sub_questions", []),
        iteration=final_state.get("iteration", 0),
        review_score=final_state.get("review_score"),
        code_images=final_state.get("code_images", []),
        execution_log=final_state.get("execution_log", ""),
        cache_source=None,  # 重新跑流水线（多轮不写缓存）
    )


@app.post("/upload", response_model=UploadResponse, status_code=202)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    roles: str = "",
    departments: str = "",
    user: UserContext = Depends(require_admin),
) -> UploadResponse:
    """
    上传知识库文件（异步处理，**仅 admin 角色**，其他人 403）：

      curl -X POST localhost:8000/upload -H "Authorization: Bearer <token>" \
           -F "file=@本地文件.txt" \
           --data-urlencode "roles=hr,admin" \
           --data-urlencode "departments="

    roles/departments 是查询参数（逗号分隔的可见角色/部门，留空=公开文档）。
    接口只做：校验 → 落盘 → 建任务记录 → 排队后台任务，**立即返回 task_id**。
    后台任务（解析 → 切块 → 向量化 → 入库）不阻塞本次请求，
    用 GET /tasks/{task_id} 查询进度；处理完成后按权限标签可被对应角色检索到。
    """
    # 基本校验：扩展名白名单 + 文件名合法性（防止路径穿越，只取纯文件名）
    safe_name = Path(file.filename or "").name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"仅支持 {sorted(ALLOWED_EXTENSIONS)} 文件，收到: {ext or '(无扩展名)'}",
        )

    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="上传文件内容为空")
    if len(body) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件过大（>{MAX_UPLOAD_SIZE // 1024 // 1024}MB）")

    # 内容 sha256 作为 doc_id：同一文件重传幂等（后台会先清旧块再入库），
    # 也为阶段 4 的缓存失效打基础
    sha256 = hashlib.sha256(body).hexdigest()
    doc_id = sha256
    # task_id 由 create_task 生成并入库，保证响应里返回的 ID 与库内一致
    task_id = create_task(doc_id=doc_id)

    # 同步落盘（快，几十毫秒内）；时间戳前缀防同名覆盖，.part 临时文件避免半截文件
    UPLOAD_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = UPLOAD_DIR / f"{stamp}_{safe_name}"
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        part.write_bytes(body)
        shutil.move(str(part), str(dest))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 登记元信息并排队后台任务；BackgroundTasks 会在本响应返回后执行。
    # 权限标签随块元数据入库，retrieve 按 JWT 的 role/department 过滤
    create_document(doc_id=doc_id, filename=safe_name, size_bytes=len(body), sha256=sha256)
    background_tasks.add_task(
        process_upload, task_id, doc_id, str(dest), safe_name, roles, departments,
    )

    return UploadResponse(
        task_id=task_id,
        doc_id=doc_id,
        filename=safe_name,
        size_bytes=len(body),
        allowed_roles=roles,
        allowed_departments=departments,
        status="pending",
        message=f"已受理，后台异步切块入库中（可见角色: {roles or '公开'}，可见部门: {departments or '公开'}）",
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str, user: UserContext = Depends(get_current_user)) -> TaskStatusResponse:
    """查询文件处理任务的阶段/进度/状态（需登录）；任务不存在返回 404。"""
    detail = get_task_detail(task_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return TaskStatusResponse(**detail)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request,
           user: UserContext = Depends(get_current_user)) -> SearchResponse:
    """
    直接检索知识库（需登录），用于验证 RBAC：不同角色同题检索结果不同。

      curl -X POST localhost:8000/search -H "Authorization: Bearer <token>" \
           -H "Content-Type: application/json" -d '{"query":"公司制度"}'

    只返回当前用户（角色/部门）有权看到的块；admin 返回全部匹配。

    阶段5 Step 3：审计记录 query + seen_doc_ids。
    """
    t0 = time.time()
    hits = retrieve(req.query, top_k=req.top_k, role=user.role, department=user.department)
    seen = sorted({
        (h.get("metadata") or {}).get("doc_id")
        for h in hits
        if (h.get("metadata") or {}).get("doc_id")
    })
    log_audit(username=user.username, role=user.role, action="/search",
              question=req.query, seen_doc_ids=seen,
              ip=request.client.host if request.client else None,
              user_agent=request.headers.get("user-agent"),
              latency_ms=int((time.time()-t0)*1000),
              status_code=200)
    return SearchResponse(
        query=req.query,
        viewer=f"{user.role}@{user.department}",
        total=len(hits),
        hits=[
            SearchHit(
                source=h["metadata"].get("source", "未知"),
                chunk_index=h["metadata"].get("chunk_index"),
                text=h["text"][:300],
                distance=h.get("distance"),
            )
            for h in hits
        ],
    )


# ---------------------- 阶段 4 缓存管理接口 ----------------------

class CacheStatsResponse(BaseModel):
    """缓存状态：供 GET /cache/stats 返回，验收时看规模与命中。"""
    model_version: str
    ttl_seconds: int
    lru_maxsize: int
    lru_size: int
    lru_retrieve_keys: int
    lru_ask_keys: int
    redis_enabled: bool
    redis_host: Optional[str] = None


class CacheClearResponse(BaseModel):
    """手动清缓存的回报：本次删了多少 LRU / Redis 键。"""
    lru_keys: int
    redis_keys: int
    message: str


@app.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats(user: UserContext = Depends(get_current_user)) -> CacheStatsResponse:
    """
    查看缓存规模与 Redis 连通状态（需登录）。

      curl localhost:8000/cache/stats -H "Authorization: Bearer <token>"
    """
    return CacheStatsResponse(**cache_mgr.stats())


@app.delete("/cache/clear", response_model=CacheClearResponse)
def cache_clear(user: UserContext = Depends(require_admin)) -> CacheClearResponse:
    """
    手动清空全部缓存（**仅 admin**）——清一级 LRU 全部 + 二级 Redis rag:*/ask:* 前缀。

      curl -X DELETE localhost:8000/cache/clear -H "Authorization: Bearer <admin_token>"

    文件更新本身会按 doc_id 精确失效相关缓存，无需手动清；此接口用于排查
    脏数据 / 切换模型版本后强制全量失效的场景。
    """
    deleted = cache_mgr.clear_all()
    return CacheClearResponse(
        lru_keys=deleted["lru_keys"],
        redis_keys=deleted["redis_keys"],
        message=f"已清空 LRU {deleted['lru_keys']} 个 + Redis {deleted['redis_keys']} 个键",
    )


# ────────────────────────── 阶段5 Step 3：审计日志查询 ──────────────────────────

@app.get("/audit/logs", response_model=List[AuditLogEntry])
def get_audit_logs(
    username: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
    user: UserContext = Depends(require_admin),
) -> List[AuditLogEntry]:
    """
    查审计日志（**仅 admin**）：

      curl "localhost:8000/audit/logs?action=/ask&limit=20" -H "Authorization: Bearer <admin>"

    支持按 username / action 过滤；按 id 倒序返回最近 limit 条（默认 100，最多 1000）。
    """
    logs = query_logs(username=username, action=action, limit=limit)
    return [AuditLogEntry(**row) for row in logs]


# ────────────────────────── 阶段5 Step 2：用户管理 ──────────────────────────

@app.get("/users", response_model=List[UserResponse])
def list_users_route(user: UserContext = Depends(require_admin)) -> List[UserResponse]:
    """列全部用户（**仅 admin**，不含 password_hash）。"""
    return [UserResponse(**u) for u in list_users()]


@app.post("/users", response_model=CreateUserResponse, status_code=201)
def create_user_route(req: CreateUserRequest,
                      user: UserContext = Depends(require_admin)) -> CreateUserResponse:
    """
    创建用户（**仅 admin**）：

      curl -X POST localhost:8000/users -H "Authorization: Bearer <admin>" \
           -H "Content-Type: application/json" \
           -d '{"username":"carol","password":"c123","role":"hr","department":"finance"}'

    密码用 bcrypt 哈希；username 冲突 → 409。
    """
    pwd_hash = hash_password(req.password)
    ok = create_user(req.username, pwd_hash, req.role, req.department)
    if not ok:
        raise HTTPException(status_code=409, detail=f"用户名已存在: {req.username}")
    _log("创建用户", actor=user.username, new_user=req.username, role=req.role)
    log_audit(username=user.username, role=user.role, action="/users",
              question=f"create {req.username}", status_code=201)
    return CreateUserResponse(username=req.username, role=req.role, department=req.department)


@app.patch("/users/{username}", response_model=UpdateUserResponse)
def update_user_route(username: str, req: UpdateUserRequest,
                      user: UserContext = Depends(require_admin)) -> UpdateUserResponse:
    """
    改密/改角色（**仅 admin**）。所有字段可选，只更新传入的。

      curl -X PATCH localhost:8000/users/bob -H "Authorization: Bearer <admin>" \
           -H "Content-Type: application/json" -d '{"role":"lead","password":"newpass"}'
    """
    fields = {}
    if req.password is not None:
        fields["password_hash"] = hash_password(req.password)
    if req.role is not None:
        fields["role"] = req.role
    if req.department is not None:
        fields["department"] = req.department
    if req.is_active is not None:
        fields["is_active"] = req.is_active
    if not fields:
        raise HTTPException(status_code=400, detail="没有要更新的字段")
    ok = update_user(username, **fields)
    if not ok:
        raise HTTPException(status_code=404, detail=f"用户不存在: {username}")
    _log("更新用户", actor=user.username, target=username, fields=list(fields.keys()))
    log_audit(username=user.username, role=user.role, action="/users",
              question=f"update {username}", status_code=200)
    return UpdateUserResponse(username=username, updated=list(fields.keys()))


# ────────────────────────── 阶段5 Step 5：多轮对话上下文 ──────────────────────────

@app.get("/sessions/{session_id}")
def get_session(session_id: str, user: UserContext = Depends(get_current_user)) -> dict:
    """
    看对话历史（需登录，仅本人或 admin）：

      curl http://127.0.0.1:8000/sessions/sess1 -H "Authorization: Bearer <token>"
    """
    history = cache_mgr.get_chat(session_id)
    return {"session_id": session_id, "total": len(history), "history": history}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str, user: UserContext = Depends(get_current_user)) -> dict:
    """
    清空对话历史（需登录）：

      curl -X DELETE http://127.0.0.1:8000/sessions/sess1 -H "Authorization: Bearer <token>"
    """
    cleared = cache_mgr.clear_chat(session_id)
    return {
        "session_id": session_id,
        "cleared": cleared,
        "message": "历史已清空" if cleared else "无历史可清（session 不存在或已空）",
    }


# ────────────────────────── 阶段5 Step 7：SSE 流式 + HITL ──────────────────────────

class AskStreamRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                         description="研究问题，如：猫有哪些生活习性？")
    hitl: bool = Field(False, description="True=planner 后暂停等待用户确认子问题（HITL 检查点）")


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="/ask_stream 返回的 thread_id")
    sub_questions: List[str] = Field(..., description="用户编辑/确认后的子问题列表")


@app.post("/ask_stream")
def ask_stream(req: AskStreamRequest, request: Request,
               user: UserContext = Depends(get_current_user)):
    """
    SSE 流式执行研究流水线，前端实时收到每个节点的进度事件。

    需要登录。不走缓存（流式无法缓存）。支持 X-Session-Id header 开多轮对话。
    hitl=true 时 planner 后暂停，返回 awaiting_user 事件；
    用户需调 POST /ask_stream/resume 提交编辑后的子问题来恢复。

    事件格式（SSE data: JSON）：
      {"type":"start","thread_id":"...","question":"...","hitl":false}
      {"type":"node","node":"planner","label":"规划拆解","sub_questions":[...]}
      {"type":"node","node":"researcher","label":"检索资料","search_count":3,...}
      ...
      {"type":"awaiting_user","thread_id":"...","sub_questions":[...]}  ← 仅 hitl=true
      {"type":"done","thread_id":"...","final_report":"...","iteration":2,...}

    curl -N -X POST http://127.0.0.1:8000/ask_stream \\
      -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \\
      -d '{"question":"猫的生活习性","hitl":false}'
    """
    import json as _json

    t0 = time.time()
    session_id = request.headers.get("x-session-id")
    history = cache_mgr.get_chat(session_id) if session_id else []
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")

    def generate():
        final_report = ""
        try:
            for sse_chunk in run_research_stream(
                question=req.question,
                role=user.role,
                department=user.department,
                history=history,
                session_id=session_id,
                hitl=req.hitl,
            ):
                yield sse_chunk
                # 从 done 事件提取 final_report（用于审计 + 会话历史）
                if '"type"' in sse_chunk and '"done"' in sse_chunk:
                    try:
                        data = _json.loads(sse_chunk.replace("data: ", "").strip())
                        final_report = data.get("final_report", "")
                    except Exception:
                        pass
        except Exception as e:
            err = _json.dumps({"type": "error", "message": str(e)},
                              ensure_ascii=False)
            yield f"data: {err}\n\n"

        # 流式完成后：审计日志 + 会话历史追加（即使出错也记录）
        latency_ms = int((time.time() - t0) * 1000)
        log_audit(username=user.username, role=user.role, action="/ask_stream",
                  question=req.question, ip=ip, user_agent=ua,
                  latency_ms=latency_ms, status_code=200 if final_report else 500)
        if final_report and session_id:
            cache_mgr.append_chat(session_id, "user", req.question)
            cache_mgr.append_chat(session_id, "assistant", final_report)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 不缓冲
        },
    )


@app.post("/ask_stream/resume")
def resume_stream(req: ResumeRequest, request: Request,
                 user: UserContext = Depends(get_current_user)):
    """
    HITL 恢复：用户编辑/确认子问题后，继续流式执行。

    接收 /ask_stream（hitl=true）返回的 thread_id 和用户编辑后的子问题，
    更新 graph state 后继续从中断点流式执行。
    支持 X-Session-Id header：流式完成后把原始问题 + 最终报告追加到会话历史。

    curl -N -X POST http://127.0.0.1:8000/ask_stream/resume \\
      -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \\
      -H "X-Session-Id: sess1" \\
      -d '{"thread_id":"run-123","sub_questions":["子问题1","子问题2"]}'
    """
    import json as _json
    session_id = request.headers.get("x-session-id")

    def generate():
        final_report = ""
        original_question = ""
        for sse_chunk in resume_research_stream(
            thread_id=req.thread_id,
            user_sub_questions=req.sub_questions,
        ):
            yield sse_chunk
            if '"type"' in sse_chunk and '"done"' in sse_chunk:
                try:
                    data = _json.loads(sse_chunk.replace("data: ", "").strip())
                    final_report = data.get("final_report", "")
                except Exception:
                    pass

        # HITL resume 完成后：把原始问题 + 最终报告追加到会话历史
        if final_report and session_id:
            # 从 graph state 取原始问题（resume 时 graph 已有 research_question）
            try:
                from main import _get_streaming_graph
                graph = _get_streaming_graph()
                state = graph.get_state({"configurable": {"thread_id": req.thread_id}})
                original_question = state.values.get("research_question", "")
            except Exception:
                pass
            if original_question:
                cache_mgr.append_chat(session_id, "user", original_question)
            cache_mgr.append_chat(session_id, "assistant", final_report)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
