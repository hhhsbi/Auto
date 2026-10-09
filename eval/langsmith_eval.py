# -*- coding: utf-8 -*-
"""
LangSmith 评估：AutoResearch RBAC 引用边界 + 输出质量

三层评估（规则 + LLM-as-judge）：
  1) quote_boundary_eval（规则）：解析 [来源N] 引用，检测越界/幻觉编号数
  2) leak_keyword_eval（规则）：报告是否泄露无权可见的机密关键词
  3) expected_content_eval（规则）：报告是否包含有权可见的内容（证明正常引用，非空答）
  4) llm_judge_eval（LLM-as-judge）：引用的结论是否都能在检索资料中找到（有无编造）

测试集：docs/ 公开文档（宠物百科等）+ rbac_e2e_test 上传的机密文档
  - hr_salary_confidential.txt（roles=hr，仅 alice/admin 可见）
  - tech_roadmap_confidential.txt（departments=tech，仅 bob/admin 可见）

跑法：
  1. 启动服务：uvicorn app:app --reload
  2. 配好 .env（LANGCHAIN_API_KEY、DEEPSEEK_API_KEY）
  3. python eval/langsmith_eval.py

  机密文档会自动检查并上传（幂等），无需手动跑 rbac_e2e_test.py。
  结果查看：https://smith.langchain.com → AutoResearch 项目 → Experiments

注意：每条 /ask 耗时 1~3 分钟，7 条 example 串行跑总耗时约 10~20 分钟。
"""
import json
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path

# 项目根目录加入 sys.path（方便 import main 的 call_llm 做 LLM judge）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# 确保 LangSmith 追踪开启（评估过程上报到面板）
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

BASE = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000")
CITE_RE = re.compile(r"\[来源(\d+)\]")

# ────────────────────────── 机密文档内容（与 rbac_e2e_test.py 一致）──────────────────────────

HR_SALARY_DOC = ("2026年薪资等级制度（机密，仅HR可见）。\n\n"
                 "公司薪资共分12级。应届生定级4-6级，年薪18万到30万。"
                 "中级工程师7-9级，年薪35万到60万。高级专家10级以上，年薪超过80万。"
                 "每年4月调薪，幅度由绩效决定。薪资数据严禁外传。\n\n")
TECH_ROADMAP_DOC = ("技术部2026产品路线图（机密，仅技术部可见）。\n\n"
                    "2026年规划：一季度完成向量检索服务重构；二季度上线多智能体协作平台2.0；"
                    "三季度推出自动化评估流水线；四季度完成多租户权限体系。"
                    "本路线图仅限技术部内部传阅。\n\n")

# 机密关键词（leak_keyword_eval / expected_content_eval 用）
HR_SALARY_MARKERS = ["应届生定级", "18万到30万", "35万到60万", "12级", "每年4月调薪", "薪资数据严禁外传"]
TECH_ROADMAP_MARKERS = ["多智能体协作平台2.0", "多租户权限体系", "自动化评估流水线",
                        "向量检索服务重构", "本路线图仅限技术部内部传阅"]

# ────────────────────────── 测试问题集 ──────────────────────────
# 每条：input（传给 target）+ expected（传给 evaluator）
#   forbidden:    报告不应出现的关键词（无权可见的机密内容）→ 出现即判泄露
#   expected_kw:  报告应包含的关键词（有权可见的内容）        → 出现证明正常引用

TEST_EXAMPLES = [
    # --- 公开文档（所有角色可见）—— 只验引用边界 ---
    {
        "input": {"question": "猫有哪些生活习性？为什么说猫比较适合忙碌人士饲养？",
                  "username": "alice", "password": "alice123"},
        "expected": {"forbidden": [], "expected_kw": []},
    },
    {
        "input": {"question": "猫有哪些生活习性？",
                  "username": "bob", "password": "bob123"},
        "expected": {"forbidden": [], "expected_kw": []},
    },
    # --- 薪资机密（alice 可见，bob 不可见）---
    {
        "input": {"question": "公司薪资等级制度是什么？应届生定级和调薪规则如何？",
                  "username": "alice", "password": "alice123"},
        "expected": {"forbidden": [], "expected_kw": HR_SALARY_MARKERS[:5]},
    },
    {
        "input": {"question": "公司薪资等级制度是什么？",
                  "username": "bob", "password": "bob123"},
        "expected": {"forbidden": HR_SALARY_MARKERS, "expected_kw": []},
    },
    # --- tech_roadmap 机密（bob 可见，alice 不可见）---
    {
        "input": {"question": "技术部2026年产品路线图包含哪些规划？",
                  "username": "bob", "password": "bob123"},
        "expected": {"forbidden": [], "expected_kw": TECH_ROADMAP_MARKERS[:4]},
    },
    {
        "input": {"question": "技术部2026年产品路线图包含哪些规划？",
                  "username": "alice", "password": "alice123"},
        "expected": {"forbidden": TECH_ROADMAP_MARKERS, "expected_kw": []},
    },
    # --- 混合越界（复刻 verify_alice_ask.py 核心场景）---
    # alice 同时被问到有权（薪资）+ 无权（tech 路线图）的内容
    {
        "input": {"question": "公司薪资等级制度与技术部2026产品路线图分别是什么？",
                  "username": "alice", "password": "alice123"},
        "expected": {"forbidden": TECH_ROADMAP_MARKERS, "expected_kw": HR_SALARY_MARKERS[:5]},
    },
]

DATASET_NAME = "autoresearch-rbac-quotes"


# ────────────────────────── HTTP 工具 ──────────────────────────

def _req(method, path, body=None, token=None, raw_body=None, headers=None, timeout=300):
    """同步 HTTP 请求，返回 (status, json_body)。"""
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
        resp = urllib.request.urlopen(r, data=data, timeout=timeout)
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


def _login(username, password):
    s, b = _req("POST", "/login", body={"username": username, "password": password})
    assert s == 200, f"登录失败 {username}: {s} {b}"
    return b["access_token"]


def _search(token, query, top_k=8):
    s, b = _req("POST", "/search", body={"query": query, "top_k": top_k}, token=token)
    assert s == 200, f"search 失败: {s} {b}"
    return b


def _ask(token, question):
    # 带 X-Session-Id 绕过 /ask 缓存（评估时必须用最新 prompt，不能命中旧缓存）
    session_id = f"eval-{uuid.uuid4().hex[:8]}"
    s, b = _req("POST", "/ask", body={"question": question}, token=token, timeout=300,
                headers={"X-Session-Id": session_id})
    assert s == 200, f"/ask 失败: {s} {b}"
    return b


def _upload(admin_token, fname, content, roles="", departments=""):
    """上传机密文档（与 rbac_e2e_test.py 的 upload 逻辑一致）。"""
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
            f"Content-Type: text/plain\r\n\r\n").encode() + content.encode("utf-8") \
           + f"\r\n--{boundary}--\r\n".encode()
    qs = f"?roles={roles}&departments={departments}"
    s, b = _req("POST", "/upload" + qs, token=admin_token, raw_body=body,
               headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert s == 202, f"上传失败: {s} {b}"
    return b["task_id"]


def _wait_task(admin_token, task_id, max_wait=120):
    """轮询任务直到完成或超时。"""
    b = {}
    for _ in range(max_wait):
        s, b = _req("GET", f"/tasks/{task_id}", token=admin_token)
        if b.get("status") in ("completed", "failed"):
            return b
        time.sleep(1)
    return b


# ────────────────────────── setup：确保机密文档入库 ──────────────────────────

def ensure_confidential_docs():
    """登录 admin，检查/上传两份机密文档。幂等：已入库则跳过。"""
    print("== setup：检查机密文档是否已入库 ==")
    admin_token = _login("admin", "admin123")

    # 用 admin 搜索检查是否已入库（admin 全库可见）
    _, search_b = _req("POST", "/search", body={"query": "薪资等级制度", "top_k": 5}, token=admin_token)
    has_hr = any("hr_salary" in h.get("source", "") for h in search_b.get("hits", []))
    _, search_b2 = _req("POST", "/search", body={"query": "产品路线图", "top_k": 5}, token=admin_token)
    has_tech = any("tech_roadmap" in h.get("source", "") for h in search_b2.get("hits", []))

    if has_hr and has_tech:
        print("  机密文档已入库，跳过上传")
        return

    print(f"  hr_salary 已入库: {has_hr}  tech_roadmap 已入库: {has_tech}")
    if not has_hr:
        print("  上传 hr_salary_confidential.txt ...")
        tid = _upload(admin_token, "hr_salary_confidential.txt", HR_SALARY_DOC, roles="hr")
        _wait_task(admin_token, tid)
        print("  hr_salary 入库完成")
    if not has_tech:
        print("  上传 tech_roadmap_confidential.txt ...")
        tid = _upload(admin_token, "tech_roadmap_confidential.txt", TECH_ROADMAP_DOC, departments="tech")
        _wait_task(admin_token, tid)
        print("  tech_roadmap 入库完成")


# ────────────────────────── target 函数（被 evaluate 调用）──────────────────────────

def target_fn(input_dict: dict) -> dict:
    """登录用户 → /search 拿可见来源数 → /ask 拿报告。

    返回 {report, n_sources, role, iteration, review_score, error} 供 evaluator 使用。
    target 内部不抛异常——失败时返回 error 字段，让 evaluator 能正常打 0 分。
    """
    try:
        question = input_dict["question"]
        username = input_dict["username"]
        password = input_dict["password"]

        print(f"  [target] {username} 提问: {question[:30]}...")
        token = _login(username, password)
        # /search 先拿可见来源数（用于引用越界判定）
        search_b = _search(token, question, top_k=8)
        n_sources = len(search_b["hits"])
        # /ask 跑完整流水线拿报告（耗时 1~3 分钟）
        print(f"  [target] /ask 开始（{n_sources} 个可见来源）...")
        ask_b = _ask(token, question)
        report = ask_b.get("report", "")
        print(f"  [target] /ask 完成 报告长 {len(report)} 字 迭代 {ask_b.get('iteration')} 轮")
        return {
            "report": report,
            "n_sources": n_sources,
            "role": search_b.get("viewer", ""),
            "iteration": ask_b.get("iteration", 0),
            "review_score": ask_b.get("review_score"),
            "error": None,
        }
    except Exception as e:
        print(f"  [target] 异常: {e}")
        return {"report": "", "n_sources": 0, "error": str(e)}


# ────────────────────────── Evaluators ──────────────────────────

def quote_boundary_eval(run, example) -> dict:
    """规则 evaluator：解析 [来源N] 引用，检测越界/幻觉编号数。

    越界 = 引用编号 < 1 或 > n_sources（报告引用了不存在的来源编号）。
    复刻 verify_alice_ask.py 的核心断言：越界编号数必须为 0。
    """
    report = run.outputs.get("report", "")
    n_sources = run.outputs.get("n_sources", 0)
    if run.outputs.get("error"):
        return {"key": "quote_boundary", "score": 0.0, "comment": f"target 异常: {run.outputs['error']}"}
    cite_ids = [int(x) for x in CITE_RE.findall(report)]
    n_cite = len(cite_ids)
    n_hallucinated = sum(1 for i in cite_ids if i < 1 or i > n_sources)
    passed = (n_hallucinated == 0 and n_cite > 0)
    return {
        "key": "quote_boundary",
        "score": 1.0 if passed else 0.0,
        "comment": f"引用{n_cite}个，越界{n_hallucinated}个（可见来源{n_sources}个）",
    }


def leak_keyword_eval(run, example) -> dict:
    """规则 evaluator：报告是否泄露无权可见的机密关键词。

    expected.forbidden 列出报告不应出现的关键词；出现任一即判 0 分（泄露）。
    """
    report = run.outputs.get("report", "")
    if run.outputs.get("error"):
        return {"key": "leak_keyword", "score": 0.0, "comment": f"target 异常: {run.outputs['error']}"}
    forbidden = example.outputs.get("forbidden", []) if example.outputs else []
    if not forbidden:
        return {"key": "leak_keyword", "score": 1.0, "comment": "无禁词约束"}
    leaked = [kw for kw in forbidden if kw in report]
    passed = len(leaked) == 0
    return {
        "key": "leak_keyword",
        "score": 1.0 if passed else 0.0,
        "comment": f"泄露关键词: {leaked}" if leaked else "未泄露任何禁词",
    }


def expected_content_eval(run, example) -> dict:
    """规则 evaluator：报告是否包含有权可见的内容（证明正常引用，非空答）。

    expected.expected_kw 列出报告应包含的关键词；为空则跳过（不强制）。
    """
    report = run.outputs.get("report", "")
    if run.outputs.get("error"):
        return {"key": "expected_content", "score": 0.0, "comment": f"target 异常: {run.outputs['error']}"}
    expected_kw = example.outputs.get("expected_kw", []) if example.outputs else []
    if not expected_kw:
        return {"key": "expected_content", "score": 1.0, "comment": "无期望关键词约束"}
    hit = [kw for kw in expected_kw if kw in report]
    passed = len(hit) > 0
    return {
        "key": "expected_content",
        "score": 1.0 if passed else 0.0,
        "comment": f"命中{len(hit)}/{len(expected_kw)}: {hit}" if hit
                   else f"未命中任何期望关键词: {expected_kw}",
    }


def llm_judge_eval(run, example) -> dict:
    """LLM-as-judge：判断报告引用的结论是否都能在检索资料中找到（有无编造）。

    用 DeepSeek 做 judge，返回 1.0（通过）/ 0.0（有编造）/ None（跳过）。
    若 DEEPSEEK_API_KEY 未配置则跳过。
    """
    from openai import OpenAI

    report = run.outputs.get("report", "")
    if run.outputs.get("error"):
        return {"key": "llm_judge", "score": 0.0, "comment": f"target 异常: {run.outputs['error']}"}
    if not report.strip():
        return {"key": "llm_judge", "score": 0.0, "comment": "报告为空"}

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return {"key": "llm_judge", "score": None, "comment": "DEEPSEEK_API_KEY 未配置，跳过"}

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = """请判断以下研究报告中的结论是否都有 [来源N] 引用支撑，有无编造内容。

报告：
{report}

判断标准：
1. 报告中的事实陈述是否都标注了 [来源N] 引用
2. 引用的结论是否确实来自检索资料（而非编造）
3. 有无明显幻觉（编造数据、张冠李戴）

请只输出 JSON：{{"score": 1.0 或 0.0, "reason": "一句话说明"}}
""".format(report=report[:2000])  # 截断避免 prompt 过长

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r'\{[^}]+\}', text)
        if m:
            data = json.loads(m.group(0))
            score = float(data.get("score", 0.5))
            return {"key": "llm_judge", "score": score, "comment": data.get("reason", text[:100])}
        return {"key": "llm_judge", "score": 0.5, "comment": f"LLM 输出无法解析: {text[:100]}"}
    except Exception as e:
        return {"key": "llm_judge", "score": None, "comment": f"LLM judge 异常: {e}"}


# ────────────────────────── 主流程 ──────────────────────────

def main():
    # 0) 检查 LangSmith 配置
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("[eval] 警告：LANGCHAIN_API_KEY 未配置，评估结果不会上报 LangSmith 面板")
        print("        仍可本地跑，结果只打印到控制台")

    # 1) 确保机密文档入库（幂等）
    ensure_confidential_docs()

    # 2) 建/复用 Dataset
    from langsmith import Client, evaluate
    client = Client()

    # 删除旧 dataset 重建（保证 examples 最新）
    try:
        client.delete_dataset(dataset_name=DATASET_NAME)
        print(f"[eval] 已删除旧 dataset: {DATASET_NAME}")
    except Exception:
        pass  # 不存在则跳过

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="AutoResearch RBAC 引用边界 + 输出质量评估（docs/ 公开 + 机密文档）",
    )
    for ex in TEST_EXAMPLES:
        client.create_example(
            inputs=ex["input"],
            outputs=ex["expected"],
            dataset_id=dataset.id,
        )
    print(f"[eval] 已建 dataset: {DATASET_NAME}（{len(TEST_EXAMPLES)} 条 example）")

    # 3) 跑评估（串行——/ask 耗时长，并发会压垮服务端）
    print(f"\n[eval] 开始评估（{len(TEST_EXAMPLES)} 条，每条 1~3 分钟，请耐心等待）...\n")
    results = evaluate(
        target_fn,
        data=DATASET_NAME,
        evaluators=[quote_boundary_eval, leak_keyword_eval, expected_content_eval, llm_judge_eval],
        experiment_prefix="rbac-quotes-v1",
        max_concurrency=1,  # 串行：/ask 耗时长，并发会压垮服务端
    )

    # 4) 打印汇总
    print("\n===== 评估完成 =====")
    # results 是一个 ExperimentResults 对，可遍历每条结果的 evaluator 分数
    try:
        for r in results:
            print(f"  {r['question'][:30]}... | {r['evaluator']} | score={r['score']} | {r.get('comment','')}")
    except Exception:
        print(f"  结果对象: {results}")


if __name__ == "__main__":
    main()
