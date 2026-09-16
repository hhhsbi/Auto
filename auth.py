"""
auth.py —— JWT 认证与 RBAC 权限（阶段 3 + 阶段5 Step 2 重构）

职责：
  1) /login 的用户校验 + JWT 签发（claims 带 role / department）
  2) FastAPI 依赖 get_current_user / require_admin：校验 Bearer Token，
     无效/缺失返回 401，角色不足返回 403

阶段5 Step 2 改造：
  - 用户从 SQLite users 表查（替代原来写死的 _USERS 字典）
  - 密码用 bcrypt 哈希（替代原来的 sha256）——bcrypt 慢哈希抗暴力破解
  - 新增 hash_password / verify_password 工具函数，供 /users 接口创建/改密用
  - 用户 CRUD 函数从 ingest 借（ingest.create_user/list_users/update_user）

JWT 密钥：优先环境变量；开发环境给默认值并提示。
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# 用户表 CRUD（ingest.py 阶段5 Step 1 新增）
from ingest import get_user_by_username

load_dotenv()

# JWT 签名密钥：优先环境变量；开发环境给默认值并提示
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    JWT_SECRET = "dev-secret-DO-NOT-USE-IN-PRODUCTION"
    print("[auth] 警告：未配置 JWT_SECRET，使用开发默认密钥（生产环境请在 .env 配置）")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))


# ────────────────────────── bcrypt 工具 ──────────────────────────

def hash_password(password: str) -> str:
    """bcrypt 哈希密码（供 /users 创建/改密用）。返回 60 字符串。"""
    import bcrypt
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """验密码；bcrypt.checkpw 自带恒定时间比较，抗时序攻击。"""
    import bcrypt
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False  # 哈希格式非法


# ────────────────────────── 用户上下文 ──────────────────────────

class UserContext(BaseModel):
    """当前登录用户的上下文，随请求在依赖链里传递，最终进入检索层做权限过滤。"""
    username: str
    role: str
    department: str


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    """POST /users 创建用户的请求体（admin only）。"""
    username: str
    password: str
    role: str
    department: str


class UpdateUserRequest(BaseModel):
    """PATCH /users/{username} 改密/改角色的请求体（admin only）；所有字段可选。"""
    password: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[int] = None


def authenticate(username: str, password: str) -> Optional[UserContext]:
    """
    校验用户名密码（从 SQLite users 表查 + bcrypt 验）。

    成功返回 UserContext，失败（用户不存在/密码错/账号停用）返回 None。
    """
    u = get_user_by_username(username)
    if not u:
        return None
    if not u.get("is_active", 1):
        return None  # 账号停用
    if not verify_password(password, u["password_hash"]):
        return None
    return UserContext(
        username=u["username"],
        role=u["role"],
        department=u["department"],
    )


def create_access_token(user: UserContext) -> str:
    """签发 JWT，claims 带 role / department（检索层要用它们做 where 过滤）。"""
    payload = {
        "sub": user.username,
        "role": user.role,
        "department": user.department,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# HTTPBearer(auto_error=False)：没带 Token 时不自动抛 401，由我们统一处理，
# 便于区分"没带 Token"（401）和"带了但角色不对"（403）
_bearer = HTTPBearer(auto_error=False)


def get_current_user(cred: HTTPAuthorizationCredentials = Depends(_bearer)) -> UserContext:
    """FastAPI 依赖：校验 Bearer JWT，解析出用户上下文。"""
    if cred is None or not cred.credentials:
        raise HTTPException(status_code=401, detail="未登录：请在 Authorization 头携带 Bearer Token（先 POST /login 获取）")
    try:
        payload = jwt.decode(cred.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token 无效")
    return UserContext(
        username=payload.get("sub", ""),
        role=payload.get("role", ""),
        department=payload.get("department", ""),
    )


def require_admin(user: UserContext = Depends(get_current_user)) -> UserContext:
    """FastAPI 依赖：要求 admin 角色，否则 403（越权访问）。"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail=f"权限不足：该操作需要 admin 角色（当前角色 {user.role or '未知'}）")
    return user
