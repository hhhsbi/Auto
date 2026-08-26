# =====================================================================
# evaluate.py —— AutoResearch 评估脚本（可复现）
#
# 干什么：
#   1) 对固定测试集逐题运行流水线，拿到最终报告和检索结果。
#   2) 计算两个核心指标：
#        · 引用准确率 Citation Accuracy（存在率 + 支撑率）
#        · 答案质量 Answer Quality（LLM 按 4 维 rubric 打分）
#   3) 输出每题明细 CSV + 汇总表，用于「优化前 vs 优化后」对比。
#
# 运行前提：
#   1) 已修复 main.py（call_llm 的 bug、retrieve 换成真实 RAG）。
#   2) docs/ 已放好知识库文档并完成索引。
#   运行： python eval/evaluate.py
# =====================================================================
import os
import sys
import re
import json
import csv
from pathlib import Path

# 让脚本能 import 项目根目录的 main，以及本目录的 test_questions
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv
from openai import OpenAI

from test_questions import TEST_QUESTIONS
from main import build_graph

load_dotenv()

# 用 DeepSeek 充当"裁判模型"，做支撑性判断和答案质量打分
JUDGE = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
JUDGE_MODEL = "deepseek-chat"


# =====================================================================
# 1. 运行流水线
# =====================================================================
def run_pipeline(question: str) -> dict:
    """对一个问题运行整条 LangGraph 流水线，返回最终状态。"""
    graph = build_graph()
    initial_state = {
        "research_question": question,
        "sub_questions": [],
        "search_results": [],
        "draft": "",
        "review_feedback": None,
        "iteration": 0,
        "final_report": None,
        "code_outputs": [],
        "code_images": [],
        "execution_log": "",
        "review_score": None,
    }
    return graph.invoke(initial_state)


# =====================================================================
# 2. 引用提取与准确率
# =====================================================================
CITE_RE = re.compile(r"\[来源(\d+)\]")

def extract_citation_ids(report: str) -> list:
    """提取报告中所有 [来源N] 形式的引用编号，例如 [来源1]、[来源3]。"""
    return [int(x) for x in CITE_RE.findall(report or "")]


def judge_support(report: str, search_results: list, valid_ids: list) -> int:
    """用裁判模型判断每个引用是否被对应资料支撑，返回"被支撑"的引用数。"""
    if not valid_ids:
        return 0
    # 构造只含被引用编号的资料列表，减少 token
    src_lines = "\n".join(
        f"[来源{i}] {search_results[i - 1].get('text', '')[:200]}"
        for i in sorted(set(valid_ids))
    )
    prompt = f"""请逐条判断报告中每个引用 [来源N] 是否真的被对应编号的资料内容支撑。

报告：
{report}

检索资料：
{src_lines}
只输出一个 JSON 对象，键是引用编号（字符串），值是 true（被支撑）或 false（不被支撑）。
例如：{{"1": true, "2": false}}
"""
    resp = JUDGE.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    verdicts = json.loads(m.group()) if m else {}
    return sum(1 for i in valid_ids if str(i) in verdicts and verdicts[str(i)] is True)


def citation_accuracy(report: str, search_results: list) -> dict:
    """
    引用准确率（核心指标）。

    定义：
      · 存在率 existence_rate = 引用编号真实存在的个数 / 总引用数
          —— 编号越界（引用了一个不存在的来源）就是"幻觉引用"。
      · 支撑率 support_rate    = 内容被资料支撑的引用数 / 总引用数
          —— 由裁判模型判断"这句话是否被第 n 条资料支撑"。
      · 综合准确率 accuracy    = 被正确支撑的引用数 / 总引用数
          —— 不存在的引用直接视为不被支撑，所以它是"存在且支撑"的占比。

    返回包含各子指标和总数，方便后续汇总。
    """
    ids = extract_citation_ids(report)
    if not ids:
        # 没有任何引用：无法定义准确率，记 None（可视为"无幻觉"，但不算成绩）
        return {"total": 0, "exist": 0, "supported": 0,
                "existence_rate": None, "support_rate": None, "accuracy": None}

    n = len(search_results)
    valid_ids = [i for i in ids if 1 <= i <= n]          # 编号真实存在的引用
    exist = len(valid_ids)
    supported = judge_support(report, search_results, valid_ids)  # 被支撑的引用数

    return {
        "total": len(ids),
        "exist": exist,
        "supported": supported,
        "existence_rate": exist / len(ids),
        "support_rate": supported / len(ids),
        "accuracy": supported / len(ids),   # 综合：存在且支撑 / 总引用
    }


# =====================================================================
# 3. 答案质量（LLM 按 rubric 打分，半自动）
# =====================================================================
def answer_quality(question: str, report: str, sub_questions: list, ground_truth: list) -> dict:
    """
    答案质量：裁判模型按 4 个维度打分，每维 1-5 分。
    关键：把 ground_truth（参考答案关键事实）传给裁判，让它对照核对"准确性"，
    否则裁判只能看报告完不完整、无法发现编造，导致一律满分、没有区分度。
    返回各维得分 + 总分（满分 20） + 归一化到 0-100 的分。
    """
    gt = "\n".join(f"- {g}" for g in (ground_truth or [])) or "（无参考答案，按常识判断）"
    prompt = f"""请作为评审专家，对下面的报告按 4 个维度打分（每维 1-5 分，5 分最好）。

研究问题：{question}
应覆盖的子问题：{sub_questions}

参考答案关键事实（用于核对准确性）：
{gt}

报告：
{report}

评分维度：
1. 完整性：是否覆盖了所有子问题？
2. 准确性：报告陈述是否与"参考答案关键事实"一致、有无编造错误？（逐条对照核对）
3. 可读性：结构是否清晰、逻辑是否连贯？
4. 诚实性：对无法回答的内容是否明确说明"无法确认"，而非编造？

只输出一个 JSON：{{"完整性": 分, "准确性": 分, "可读性": 分, "诚实性": 分}}
"""
    resp = JUDGE.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    scores = json.loads(m.group()) if m else {}
    total = sum(float(v) for v in scores.values()) if scores else 0.0
    return {
        "dims": scores,
        "total": total,               # 满分 20
        "normalized": total / 20 * 100,   # 归一化到 0-100
    }


# =====================================================================
# 4. 主流程
# =====================================================================
def main():
    rows = []
    print("=" * 60)
    print("AutoResearch 评估开始")
    print("=" * 60)

    for tq in TEST_QUESTIONS:
        qid = tq["id"]
        question = tq["question"]
        print(f"\n[{qid}] 运行：{question[:40]}...")

        # 跑流水线
        state = run_pipeline(question)
        report = state.get("final_report") or state.get("draft", "")
        search_results = state.get("search_results", [])
        sub_questions = state.get("sub_questions", [])

        # 算指标
        cit = citation_accuracy(report, search_results)
        qa = answer_quality(question, report, sub_questions, tq.get("ground_truth", []))

        rows.append({
            "id": qid,
            "category": tq["category"],
            "question": question,
            "citation_total": cit["total"],
            "citation_accuracy": (f"{cit['accuracy']:.2%}" if cit["accuracy"] is not None else "N/A"),
            "existence_rate": (f"{cit['existence_rate']:.2%}" if cit["existence_rate"] is not None else "N/A"),
            "support_rate": (f"{cit['support_rate']:.2%}" if cit["support_rate"] is not None else "N/A"),
            "quality_total": qa["total"],
            "quality_normalized": f"{qa['normalized']:.1f}",
            "iteration": state.get("iteration", 0),
        })

        print(f"      引用准确率={rows[-1]['citation_accuracy']}  "
              f"答案质量={rows[-1]['quality_normalized']}/100  "
              f"迭代={rows[-1]['iteration']}")

    # 写 CSV 明细
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 打印汇总
    accs = [r["citation_accuracy"] for r in rows if r["citation_accuracy"] != "N/A"]
    acc_vals = [float(a.strip("%")) / 100 for a in accs]
    qa_vals = [float(r["quality_normalized"]) for r in rows]

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"平均引用准确率：{sum(acc_vals)/len(acc_vals):.2%}" if acc_vals else "平均引用准确率：N/A")
    print(f"平均答案质量（0-100）：{sum(qa_vals)/len(qa_vals):.1f}")
    print(f"明细已保存到：{csv_path}")


if __name__ == "__main__":
    main()
