# -*- coding: utf-8 -*-
"""阶段4 缓存验收（与 verify_alice_ask.py / rbac_e2e_test.py 同性质验收脚本）。

四个场景：
  1) retrieve 缓存命中：同 /search 第二次明显更快，cache_stats.lru_retrieve_keys +1
  2) /ask 缓存命中：同 question 第二次 cache_source="lru"，秒级返回
  3) 精确失效 invalidate_doc：直接调 cache API 按 doc_id 失效，重检索 miss 缓存
  4) DELETE /cache/clear 全清：清完 lru_size 归 0，重检索必然走 Chroma

跑：python verify_cache.py   （依赖服务已 uvicorn app:app 启动）
"""
import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"


def req(method, path, body=None, token=None):
    r = urllib.request.Request(BASE + path, method=method)
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        r.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        resp = urllib.request.urlopen(r, data=data, timeout=300)
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def login(u, p):
    s, b = req("POST", "/login", body={"username": u, "password": p})
    assert s == 200, f"login {u} fail: {s} {b}"
    return b["access_token"]


def search(token, q, top_k=5):
    t0 = time.time()
    s, b = req("POST", "/search", body={"query": q, "top_k": top_k}, token=token)
    return b, time.time() - t0


def stats(token):
    s, b = req("GET", "/cache/stats", token=token)
    return b


PASS = []


def ok(name, cond, detail=""):
    PASS.append((name, cond))
    print(f"  [{'通过' if cond else '失败'}] {name} {detail}")


admin_t = login("admin", "admin123")
alice_t = login("alice", "alice123")

# ─────────── 0. 清空缓存，确保起点干净 ───────────
print("== 0) 清空缓存（起点干净）==")
s, b = req("DELETE", "/cache/clear", token=admin_t)
print(f"  清空回报: {b}")
st = stats(admin_t)
ok("清空后 lru_size=0", st["lru_size"] == 0, f"lru_size={st['lru_size']}")

# ─────────── 1) retrieve 缓存命中 ───────────
print("\n== 1) retrieve 缓存命中（同查询第二次明显更快）==")
QUERY1 = "猫的生活习性"
b1, t1 = search(alice_t, QUERY1)
b2, t2 = search(alice_t, QUERY1)
print(f"  第一次: {t1*1000:.1f}ms  召回 {b1['total']} 条")
print(f"  第二次: {t2*1000:.1f}ms  召回 {b2['total']} 条")
ok("第二次结果与第一次一致", b1["hits"][0]["source"] == b2["hits"][0]["source"])
ok("第二次明显更快（LRU 命中）", t2 < t1 * 0.5,
   f"加速比 {t1/t2:.1f}x")
st = stats(admin_t)
ok("lru_retrieve_keys 增加（≥1）", st["lru_retrieve_keys"] >= 1,
   f"lru_retrieve_keys={st['lru_retrieve_keys']}")

# ─────────── 2) /ask 缓存命中 ───────────
print("\n== 2) /ask 缓存命中（同问题第二次 cache_source=lru）==")
QUESTION = "猫有哪些生活习性？为什么说猫比较适合忙碌人士饲养？"
print(f"  跑第一次 /ask（耗时 1~3 分钟）...")
t0 = time.time()
s, b = req("POST", "/ask", body={"question": QUESTION}, token=alice_t)
t_ask1 = time.time() - t0
assert s == 200, f"/ask 1 fail: {s} {b}"
print(f"  第一次: {t_ask1:.1f}s  cache_source={b.get('cache_source')}  report 长 {len(b['report'])} 字")
ok("第一次 cache_source=None（未命中）", b.get("cache_source") is None)

t0 = time.time()
s, b = req("POST", "/ask", body={"question": QUESTION}, token=alice_t)
t_ask2 = time.time() - t0
assert s == 200, f"/ask 2 fail: {s} {b}"
print(f"  第二次: {t_ask2*1000:.1f}ms  cache_source={b.get('cache_source')}")
ok("第二次 cache_source='lru'（命中一级）", b.get("cache_source") == "lru")
ok("第二次明显更快（秒级 vs 分钟级）", t_ask2 < t_ask1 * 0.05,
   f"加速比 {t_ask1/max(t_ask2,0.001):.1f}x")

# ─────────── 3) 精确失效 invalidate_doc（端到端：上传同内容文件触发） ───────────
print("\n== 3) 精确失效 invalidate_doc（端到端：admin 重传同内容文件触发失效）==")
# 设计：/cache/clear 起点 → alice 检索写入 LRU → admin 重传同内容文件
# → process_upload 调 delete_doc + invalidate_doc(同 doc_id) + reindex → LRU 被清
# → alice 重检索重新写入 LRU
# 观测信号用 /cache/stats.lru_retrieve_keys 数字变化（不依赖耗时——Chroma 本地
# 查询 2ms 与 LRU 命中 2ms 相当，耗时无法区分）
QUERY3 = "猫的生活习性"

# 3.1 清空缓存确保起点干净
req("DELETE", "/cache/clear", token=admin_t)
st = stats(admin_t)
lru0 = st["lru_retrieve_keys"]
print(f"  3.1 清空后 lru_retrieve_keys={lru0}")
ok("起点 lru_retrieve_keys=0", lru0 == 0)

# 3.2 第一次检索 → 写入 LRU
print("  3.2 第一次检索（写入 LRU）...")
b_q3_1, _ = search(alice_t, QUERY3)
st = stats(admin_t)
lru1 = st["lru_retrieve_keys"]
print(f"    写入后 lru_retrieve_keys={lru1}  命中 {b_q3_1['total']} 条")
ok("第一次检索写入 LRU（lru_retrieve_keys 变 1）", lru1 == 1)

# 3.3 第二次检索 → 命中 LRU，不增加 key 数
print("  3.3 第二次检索（应命中 LRU）...")
b_q3_2, _ = search(alice_t, QUERY3)
st = stats(admin_t)
lru2 = st["lru_retrieve_keys"]
print(f"    命中后 lru_retrieve_keys={lru2}")
ok("第二次命中 LRU（key 数不增加）", lru2 == lru1)

# 3.4 admin 重传 docs/宠物百科.txt（同内容 → doc_id 相同 → 触发 delete_doc+invalidate_doc+reindex）
print("  3.4 admin 重传 docs/宠物百科.txt（触发 invalidate_doc）...")
import uuid as _uuid
_BOUNDARY = _uuid.uuid4().hex
with open("docs/宠物百科.txt", "rb") as _f:
    PET_DOC_BYTES = _f.read()
# 用同名同内容文件模拟"重传"——doc_id 相同，process_upload 调 delete_doc+invalidate_doc(同 doc_id)+reindex
body = (f"--{_BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"宠物百科.txt\"\r\n"
        "Content-Type: text/plain\r\n\r\n").encode() + PET_DOC_BYTES + f"\r\n--{_BOUNDARY}--\r\n".encode()
r = urllib.request.Request(f"{BASE}/upload", method="POST", data=body)
r.add_header("Authorization", f"Bearer {admin_t}")
r.add_header("Content-Type", f"multipart/form-data; boundary={_BOUNDARY}")
try:
    resp = urllib.request.urlopen(r, timeout=60)
    upload_b = json.loads(resp.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    upload_b = json.loads(e.read().decode("utf-8"))
    raise RuntimeError(f"upload 失败: {e.code} {upload_b}")
task_id = upload_b["task_id"]
print(f"    task_id={task_id[:8]}  doc_id={upload_b['doc_id'][:12]}...")

# 等后台任务完成
for _ in range(60):
    time.sleep(1)
    s, b = req("GET", f"/tasks/{task_id}", token=admin_t)
    if b.get("status") in ("completed", "failed"):
        break
print(f"    任务状态: {b.get('status')}  入库块数={b.get('chunk_count')}")
ok("重传任务 completed", b.get("status") == "completed",
   f"status={b.get('status')}")

# 3.5 重传后看 LRU——反向索引里"猫的生活习性"query_sha1 关联了宠物百科 doc_id，
# invalidate_doc 应清掉该 LRU key
st = stats(admin_t)
lru3 = st["lru_retrieve_keys"]
print(f"  3.5 重传后 lru_retrieve_keys={lru3}（应被失效回到 0）")
ok("缓存被精确失效（lru_retrieve_keys 回 0）", lru3 == 0,
   f"实际 {lru3}")

# 3.6 alice 重检索 → 缓存 miss → 重新走 Chroma → 重新写入 LRU
print("  3.6 alice 重检索（缓存 miss，重新走 Chroma）...")
b_q3_3, _ = search(alice_t, QUERY3)
st = stats(admin_t)
lru4 = st["lru_retrieve_keys"]
print(f"    重检索后 lru_retrieve_keys={lru4}")
ok("重检索重新写入 LRU（lru_retrieve_keys 变 1）", lru4 == 1)

# ─────────── 4) DELETE /cache/clear 全清 ───────────
print("\n== 4) DELETE /cache/clear 全量清空 ==")
st_before = stats(admin_t)
print(f"  清空前 lru_size={st_before['lru_size']}")
s, b = req("DELETE", "/cache/clear", token=admin_t)
assert s == 200, f"clear fail: {s} {b}"
print(f"  清空回报: {b}")
st_after = stats(admin_t)
print(f"  清空后 lru_size={st_after['lru_size']}  redis_enabled={st_after['redis_enabled']}")
ok("清空后 lru_size=0", st_after["lru_size"] == 0)
ok("清空回报含删除计数", "lru_keys" in b and "redis_keys" in b)

# alice 看不到 admin 的清理接口（403）
s, b = req("DELETE", "/cache/clear", token=alice_t)
ok("alice 调 DELETE /cache/clear -> 403（仅 admin）", s == 403)

# ─────────── 汇总 ───────────
n_pass = sum(1 for _, c in PASS if c)
print(f"\n===== 阶段4 缓存验收：{n_pass}/{len(PASS)} 通过 =====")
