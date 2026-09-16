# -*- coding: utf-8 -*-
"""阶段3 终验：以 alice(hr) 身份跑完整 /ask，验证报告只引用其有权看的来源。

与 rbac_e2e_test.py 配对构成阶段3 验收工具链：
  - rbac_e2e_test.py：验证 RBAC 检索层（不同角色 /search 结果不同）
  - verify_alice_ask.py：验证 RBAC 贯穿到 LLM 写作层（alice /ask 报告不泄露无权内容）

每次改 RBAC 后建议重跑两个脚本作为回归验收。
"""
import json
import re
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
# 问题：同时触及 alice 有权看的薪资机密 + 无权看的 tech 路线图
QUESTION = "公司薪资等级制度与技术部2026产品路线图分别是什么？"
# hr_salary 机密正文里的特征关键词（来自 rbac_e2e_test.py 上传的内容）
HR_SALARY_MARKERS = ["应届生定级", "18万到30万", "35万到60万", "12级", "每年4月调薪"]
# tech_roadmap 机密正文里的特征关键词（alice 不应能从该文档泄露任何内容）
TECH_ROADMAP_MARKERS = ["多智能体协作平台2.0", "多租户权限体系", "自动化评估流水线",
                        "向量检索服务重构", "本路线图仅限技术部内部传阅"]
CITE_RE = re.compile(r"\[来源(\d+)\]")


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


print("== 1) alice 登录 ==")
s, b = req("POST", "/login", body={"username": "alice", "password": "alice123"})
assert s == 200, f"登录失败: {s} {b}"
tok = b["access_token"]
print(f"  alice 登录成功 role={b['role']} dept={b['department']}")

print("== 2) alice 用 /search 看同一问题能检索到哪些来源（验证可见集合）==")
s, b = req("POST", "/search", body={"query": QUESTION, "top_k": 8}, token=tok)
assert s == 200, f"search 失败: {s} {b}"
sources = [h["source"] for h in b["hits"]]
print(f"  alice 可见来源: {sources}")
sees_hr_salary = any("hr_salary" in src for src in sources)
sees_tech_roadmap = any("tech_roadmap" in src for src in sources)
print(f"  alice 看得到 hr_salary 机密? {sees_hr_salary}  (应为 True)")
print(f"  alice 看得到 tech_roadmap 机密? {sees_tech_roadmap}  (应为 False)")
assert sees_hr_salary, "alice(hr) 应能看到 hr_salary 机密文档"
assert not sees_tech_roadmap, "alice(hr) 不应看到 tech_roadmap 机密文档"
n_sources = len(sources)

print("== 3) alice 跑完整 /ask（耗时 1~3 分钟）==")
t0 = time.time()
s, b = req("POST", "/ask", body={"question": QUESTION}, token=tok)
elapsed = time.time() - t0
assert s == 200, f"/ask 失败: {s} {b}"
report = b["report"]
print(f"  /ask 完成 耗时 {elapsed:.1f}s  报告长 {len(report)} 字  迭代 {b['iteration']} 轮  评审分 {b['review_score']}")

print("\n--- 报告全文 ---")
print(report)
print("--- 报告全文结束 ---\n")

print("== 4) 报告引用分析 ==")
cite_ids = [int(x) for x in CITE_RE.findall(report)]
print(f"  报告中的引用编号: {sorted(set(cite_ids))}")
n_cite = len(cite_ids)
n_valid = sum(1 for i in cite_ids if 1 <= i <= n_sources)
n_hallucinated = n_cite - n_valid
print(f"  总引用数={n_cite}  有效(1~{n_sources})={n_valid}  幻觉(越界)={n_hallucinated}")

# 验证 1：引用无幻觉（越界编号 = 0）
assert n_hallucinated == 0, f"发现 {n_hallucinated} 个幻觉引用（越界编号）"

# 验证 2：报告不应泄露 tech_roadmap 机密的具体内容
leaked_tech = [m for m in TECH_ROADMAP_MARKERS if m in report]
print(f"  报告泄露 tech_roadmap 机密关键词: {leaked_tech}  (应为空)")
assert not leaked_tech, f"报告泄露了 tech_roadmap 机密内容: {leaked_tech}"

# 验证 3：报告应包含 hr_salary 机密的具体内容（证明 alice 有权的部分能正常生成）
hit_hr = [m for m in HR_SALARY_MARKERS if m in report]
print(f"  报告包含 hr_salary 机密关键词: {hit_hr}  (应非空)")
assert hit_hr, "报告未引用任何 hr_salary 机密内容，验证无意义"

# 验证 4：报告应包含至少一个 [来源N] 引用（否则无法谈"引用是否对"）
assert n_cite > 0, "报告无任何引用，无法验证引用真实性"

print("\n===== 终验通过：alice 报告只引用其有权看的来源 =====")
print(f"  · 引用 {n_cite} 个，越界幻觉 {n_hallucinated} 个")
print(f"  · 报告包含 {len(hit_hr)} 个 hr_salary 机密关键词（有权的部分正常引用）")
print(f"  · 报告未泄露任何 tech_roadmap 机密关键词（无权部分被隔离）")
