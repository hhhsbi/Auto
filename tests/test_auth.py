"""
test_auth.py —— JWT 认证与 RBAC 单元测试

覆盖：
  1) bcrypt hash/verify（含错密码、异常哈希）
  2) create_access_token 签发 + claims 内容
  3) JWT 解码（正常/过期/篡改）
  4) require_admin 角色检查（admin 通过，非 admin 抛 403）

不连真实 SQLite：mock get_user_by_username 返回假用户。
"""
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from auth import (
    hash_password, verify_password, create_access_token,
    UserContext, JWT_SECRET, JWT_ALGORITHM, require_admin,
)


class TestBcrypt:
    """bcrypt 哈希 + 验证。"""

    def test_hash_returns_60_chars(self):
        h = hash_password("password123")
        assert len(h) == 60
        assert h != "password123"  # 不存明文

    def test_hash_different_each_time(self):
        """bcrypt 加盐，同一密码哈希不同。"""
        h1 = hash_password("test")
        h2 = hash_password("test")
        assert h1 != h2

    def test_verify_correct_password(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("correct")
        assert verify_password("wrong", h) is False

    def test_verify_invalid_hash_returns_false(self):
        assert verify_password("pw", "not-a-valid-hash") is False

    def test_verify_empty_hash(self):
        assert verify_password("pw", "") is False


class TestCreateToken:
    """JWT 签发。"""

    def test_token_contains_claims(self):
        user = UserContext(username="alice", role="hr", department="tech")
        token = create_access_token(user)
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        assert payload["sub"] == "alice"
        assert payload["role"] == "hr"
        assert payload["department"] == "tech"
        assert "exp" in payload
        assert "iat" in payload

    def test_token_has_expiry(self):
        user = UserContext(username="bob", role="employee", department="sales")
        token = create_access_token(user)
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)
        # 过期时间在 1~3 小时后（默认 120 分钟）
        assert now < exp < now + timedelta(hours=3)

    def test_token_tampered_fails(self):
        """篡改 token 的 payload 后解码失败。"""
        user = UserContext(username="alice", role="hr", department="tech")
        token = create_access_token(user)
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(pyjwt.InvalidTokenError):
            pyjwt.decode(tampered, JWT_SECRET, algorithms=[JWT_ALGORITHM])

    def test_token_wrong_secret_fails(self):
        user = UserContext(username="alice", role="hr", department="tech")
        token = create_access_token(user)
        with pytest.raises(pyjwt.InvalidTokenError):
            pyjwt.decode(token, "wrong-secret", algorithms=[JWT_ALGORITHM])


class TestRequireAdmin:
    """require_admin 依赖：角色检查。"""

    def test_admin_passes(self):
        """admin 用户通过 require_admin。"""
        user = UserContext(username="admin", role="admin", department="it")
        result = require_admin(user)
        assert result.role == "admin"

    def test_non_admin_raises_403(self):
        """非 admin 用户被拒（403）。"""
        user = UserContext(username="alice", role="hr", department="tech")
        with pytest.raises(HTTPException) as exc:
            require_admin(user)
        assert exc.value.status_code == 403

    def test_employee_raises_403(self):
        user = UserContext(username="bob", role="employee", department="sales")
        with pytest.raises(HTTPException) as exc:
            require_admin(user)
        assert exc.value.status_code == 403


class TestAuthenticate:
    """authenticate 函数（mock 数据库）。"""

    @patch("auth.get_user_by_username")
    def test_authenticate_success(self, mock_get):
        mock_get.return_value = {
            "username": "alice",
            "password_hash": hash_password("alice123"),
            "role": "hr",
            "department": "tech",
            "is_active": 1,
        }
        from auth import authenticate
        user = authenticate("alice", "alice123")
        assert user is not None
        assert user.username == "alice"
        assert user.role == "hr"

    @patch("auth.get_user_by_username")
    def test_authenticate_wrong_password(self, mock_get):
        mock_get.return_value = {
            "username": "alice",
            "password_hash": hash_password("alice123"),
            "role": "hr",
            "department": "tech",
            "is_active": 1,
        }
        from auth import authenticate
        user = authenticate("alice", "wrong")
        assert user is None

    @patch("auth.get_user_by_username")
    def test_authenticate_user_not_found(self, mock_get):
        mock_get.return_value = None
        from auth import authenticate
        user = authenticate("nobody", "pw")
        assert user is None

    @patch("auth.get_user_by_username")
    def test_authenticate_inactive_user(self, mock_get):
        mock_get.return_value = {
            "username": "bob",
            "password_hash": hash_password("bob123"),
            "role": "employee",
            "department": "sales",
            "is_active": 0,  # 停用
        }
        from auth import authenticate
        user = authenticate("bob", "bob123")
        assert user is None
