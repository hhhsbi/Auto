# -*- coding: utf-8 -*-
"""阶段3 RBAC 端到端验收脚本（临时，跑完即删）"""
import json
import time
import uuid
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
PASS = []


def ok(name, cond, detail=""):
    PASS.append((name, cond))
    print(f"  [{'通过' if cond else '失败'}] {name} {detail}")


def req(method, path, body=None, token=None, raw_body=None, headers=None):
    r = urllib.request.Request(BASE + path, method=method)
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        h["Content-Type"] = "application/json; charset=utf-8"
    if raw_body is not None:
        data = raw_body
    for k, v in h.items():
        r.add_header(k, v)
    try:
        resp = urllib.request.urlopen(r, data=data, timeout=60)
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


def login(u, p):
    s, b = req("POST", "/login", body={"username": u, "password": p})
    assert s == 200, f"登录失败 {u}: {s} {b}"
    return b["access_token"]


print("== 1) 健康检查（无需登录）==")
s, b = req("GET", "/health")
ok("GET /health 无需 token 返回 200", s == 200, f"chunks={b.get('knowledge_chunks')}")

print("== 2) 未带 token 访问受保护接口 -> 401 ==")
s, b = req("POST", "/ask", body={"question": "测试"})
ok("POST /ask 无 token -> 401", s == 401, str(b.get("detail", ""))[:40])
s, b = req("GET", "/tasks/abc")
ok("GET /tasks 无 token -> 401", s == 401)
s, b = req("POST", "/search", body={"query": "测试"})
ok("POST /search 无 token -> 401", s == 401)

print("== 3) 登录 ==")
admin_t = login("admin", "admin123")
alice_t = login("alice", "alice123")
bob_t = login("bob", "bob123")
s, b = req("POST", "/login", body={"username": "alice", "password": "wrong"})
ok("错误密码 -> 401", s == 401)
print("  三个账号登录成功")

print("== 4) admin 上传两份受限文档，分别打标签 ==")
hr_doc = ("2026年薪资等级制度（机密，仅HR可见）。\n\n"
          "公司薪资共分12级。应届生定级4-6级，年薪18万到30万。"
          "中级工程师7-9级，年薪35万到60万。高级专家10级以上，年薪超过80万。"
          "每年4月调薪，幅度由绩效决定。薪资数据严禁外传。\n\n")
tech_doc = ("技术部2026产品路线图（机密，仅技术部可见）。\n\n"
            "2026年规划：一季度完成向量检索服务重构；二季度上线多智能体协作平台2.0；"
            "三季度推出自动化评估流水线；四季度完成多租户权限体系。"
            "本路线图仅限技术部内部传阅。\n\n")


def upload(fname, content, roles="", departments=""):
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
            f"Content-Type: text/plain\r\n\r\n").encode() + content.encode("utf-8") \
           + f"\r\n--{boundary}--\r\n".encode()
    qs = f"?roles={roles}&departments={departments}"
    s, b = req("POST", "/upload" + qs, token=admin_t, raw_body=body,
               headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert s == 202, f"上传失败: {s} {b}"
    return b["task_id"]


t1 = upload("hr_salary_confidential.txt", hr_doc, roles="hr")
t2 = upload("tech_roadmap_confidential.txt", tech_doc, departments="tech")
# 等后台入库完成
for t in (t1, t2):
    for _ in range(60):
        s, b = req("GET", f"/tasks/{t}", token=admin_t)
        if b.get("status") in ("completed", "failed"):
            break
        time.sleep(1)
    ok(f"任务 {t[:8]} 入库完成", b.get("status") == "completed",
       f"chunks={b.get('chunk_count')}")

print("== 5) 越权访问 -> 403 ==")
s, b = req("POST", "/upload?roles=hr", token=alice_t, raw_body=b"x",
           headers={"Content-Type": "application/octet-stream"})
ok("alice(hr) 上传文档 -> 403", s == 403, str(b.get("detail", ""))[:40])
s, b = req("POST", "/upload", token=bob_t, raw_body=b"x",
           headers={"Content-Type": "application/octet-stream"})
ok("bob(employee) 上传文档 -> 403", s == 403)
s, b = req("POST", "/ask", body={"question": "薪资等级"},
           token="invalid.token.here")
ok("伪造 token 访问 /ask -> 401", s == 401)

print("== 6) 同题检索，不同角色结果不同 ==")
def search(token, q):
    s, b = req("POST", "/search", token=token, body={"query": q, "top_k": 8})
    assert s == 200, f"search 失败: {s} {b}"
    return b

# 查"薪资等级"：alice(hr) 应命中 hr_salary；bob 不应命中
# 注意：测试上传的受限文档文件名是英文 hr_salary_confidential.txt，
# 故按文件名前缀 "hr_salary" / "tech_roadmap" 匹配，避免用中文匹配英文文件名造成的误判。
ra = search(alice_t, "薪资等级制度")
rb = search(bob_t, "薪资等级制度")
a_hit = any("hr_salary" in h["source"] for h in ra["hits"])
b_hit = any("hr_salary" in h["source"] for h in rb["hits"])
ok("alice(hr) 检索到薪资机密文档", a_hit, f"sources={[h['source'] for h in ra['hits']][:3]}")
ok("bob 检索不到薪资机密文档", not b_hit, f"sources={[h['source'] for h in rb['hits']][:3]}")

# 查"产品路线图"：bob(tech) 应命中 tech_roadmap；alice 不应命中
rb2 = search(bob_t, "产品路线图")
ra2 = search(alice_t, "产品路线图")
ok("bob(tech) 检索到技术路线图", any("tech_roadmap" in h["source"] for h in rb2["hits"]),
   f"sources={[h['source'] for h in rb2['hits']][:3]}")
ok("alice 检索不到技术路线图", all("tech_roadmap" not in h["source"] for h in ra2["hits"]),
   f"sources={[h['source'] for h in ra2['hits']][:3]}")

# 公开文档两边都能看到
rc = search(alice_t, "机器学习")
ok("公开文档 alice 可见", any("人工智能" in h["source"] for h in rc["hits"]))
rc2 = search(bob_t, "机器学习")
ok("公开文档 bob 可见", any("人工智能" in h["source"] for h in rc2["hits"]))

# admin 全库可见
radm = search(admin_t, "薪资等级制度")
ok("admin 全库可见（含薪资机密）", any("hr_salary" in h["source"] for h in radm["hits"]),
   f"sources={[h['source'] for h in radm['hits']][:3]}")

n_pass = sum(1 for _, c in PASS if c)
print(f"\n===== RBAC 验收：{n_pass}/{len(PASS)} 通过 =====")
