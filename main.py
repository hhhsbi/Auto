import os
import subprocess
import tempfile
import shutil
import time
import re
import sys
from typing import List, Dict, Any, Optional, Literal, TypedDict
from pathlib import Path

import openai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor
from duckduckgo_search import DDGS  # 联网搜索（DuckDuckGo）

# 导入阶段2的真实 RAG 检索（本地知识库），替代原来的 mock retrieve
from rag import retrieve, index_documents

# 项目根目录（绝对路径），保证从任何工作目录运行都能正确定位 docs/outputs。
BASE_DIR = Path(__file__).resolve().parent

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set")

# 联网搜索开关：默认关闭（国内网络下 DuckDuckGo 被墙，开启会一直失败并拖慢评估）。
# 网络可用时设环境变量 ENABLE_WEB_SEARCH=true，即可开启"本地知识库 + 联网"双路检索。
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"

# 反思循环最大迭代轮次（防止无限循环；评估时设为 1 即为"无反思"基线）
MAX_ITERATIONS = 3

# 确保本地知识库已索引（幂等：集合里已有数据则跳过；docs 目录不存在则提示）
try:
    index_documents(str(BASE_DIR / "docs"))
except FileNotFoundError:
    print("提示：./docs 目录不存在，本地检索将返回空结果。")


class ResearchState(TypedDict):
    research_question: str  # 用户原始研究问题，任务的总目标
    sub_questions: List[str]  # planner节点输出：拆解后的子问题列表
    search_results: List[Dict[str, Any]]  # researcher节点输出：检索得到的知识库片段，每个字典包含text正文、metadata元数据、distance向量相似度
    draft: str  # writer节点输出：报告草稿
    review_feedback: Optional[str]  # reviewer节点输出：评审修改意见；None代表评审通过无意见
    iteration: int  # 迭代计数器，记录评审重写轮次，用来控制最大循环次数
    final_report: Optional[str]  # 评审合格后的最终报告；None代表还没有合格定稿
    # ========== 新增：代码执行相关状态字段 ==========
    code_outputs: List[str]  # 保存代码执行全部输出：打印输出、报错信息、图表路径
    code_images: List[str]  # 保存代码生成的图片文件路径（png/jpg图表）
    execution_log: str  # 代码执行日志，记录是否执行、耗时、成功失败状态
    review_score: Optional[float]


def call_llm(prompt: str) -> str:
    client = openai.OpenAI(
        api_key=DEEPSEEK_API_KEY,              # 修复：原来误写成字符串 "DEEPSEEK_API_KEY"
        base_url="https://api.deepseek.com",   # 修复：原来误写成 deepseed.com
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],  # 修复：原来误写成 "context"
        temperature=0.3,
    )
    return response.choices[0].message.content


def execute_python_code(code: str, timeout_sec: int = 10) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script_path = tmp_path / "script.py"
        script_path.write_text(code, encoding="utf-8")
        output_dir = BASE_DIR / "outputs"
        output_dir.mkdir(exist_ok=True)
        cmd = [sys.executable, "-I", str(script_path)]  # 用当前解释器，避免系统 python 不在 PATH
        env = os.environ.copy()
        env["DEEPSEEK_API_KEY"] = "1"
        env["PYTHONPATH"] = ""
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(tmp_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check = False
            )
        except subprocess.TimeoutExpired:
            # 捕获超时异常，直接返回超时结果
            return {
                "success": False,
                "stdout": "",
                "stderr": f"超时 (>{timeout_sec}s)",
                "generated_files": [],
                "elapsed": timeout_sec
            }
        except Exception as e:
            # 其它异常（如解释器找不到）也不要让整个流程崩溃
            return {
                "success": False,
                "stdout": "",
                "stderr": f"执行出错: {e}",
                "generated_files": [],
                "elapsed": 0.0
            }
        generated = []
        for f in tmp_path.glob("*"):
            # 排除脚本本身script.py，只拿代码新生成的文件
            if f.is_file() and f.name != "script.py":
                dest = output_dir / f.name
                shutil.copy(f, dest)  # 将临时目录文件复制到outputs持久保存
                generated.append(str(dest))
        return {
            "success": result.returncode == 0,  # returncode等于0代表代码运行无报错
            "stdout": result.stdout,  # print打印输出内容
            "stderr": result.stderr,  # 报错、警告信息
            "generated_files": generated,  # 生成的全部文件路径列表
            "elapsed": time.time() - start  # 代码实际执行耗时，单位秒
        }


def web_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    联网搜索（DuckDuckGo）：把搜索结果转换成和 search_results 相同的数据结构，
    方便 writer_node 直接复用。
    每个结果包含：text 正文、metadata（source 为 "Web: 标题"、url 为网页链接）、distance。
    联网失败时返回空列表，不让整个流程崩溃。
    """
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=top_k))
    except Exception as e:
        print(f"  联网搜索失败（{e}），跳过联网检索")
        return []
    results = []
    for r in raw:
        results.append({
            "text": r.get("body", ""),
            "metadata": {
                "source": f"Web: {r.get('title', '未知')}",
                "url": r.get("href", ""),
            },
            "distance": 0.0,   # 联网结果没有向量距离，统一置 0
        })
    return results


def planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    规划节点：把用户的大研究问题拆分成多个子问题
    :param state: 全局状态
    :return: 更新state中的sub_questions字段
    """
    question = state["research_question"]
    prompt = f"请将以下研究问题拆解为 3-5 个子问题，每个一行：\n{question}"
    response = call_llm(prompt)
    # 按换行分割、过滤空行，并去掉每行开头的序号前缀（如 "1." "2、" "一、"），
    # 否则序号会被带进检索关键词，影响召回质量。
    sub_qs = []
    for line in response.split("\n"):
        line = re.sub(r'^\s*(?:\d+[\.、\)）]|[一二三四五六七八九十]+[、\.])\s*', '', line).strip()
        if line:
            sub_qs.append(line)
    return {"sub_questions": sub_qs}


def researcher_node(state: ResearchState) -> Dict[str, Any]:
    """
    调研检索节点（双路检索）：
      1) 本地知识库检索（retrieve，Chroma 向量库）
      2) 联网搜索（web_search，DuckDuckGo，由 ENABLE_WEB_SEARCH 控制）
    遍历子问题，把两类结果按"正文去重"合并进 search_results。
    :param state: 全局状态，读取 sub_questions
    :return: 更新 search_results 字段
    """
    sub_qs = state.get("sub_questions", [])
    all_results = []
    seen_texts = set()  # 按正文去重，避免本地和联网结果重复
    for q in sub_qs:
        # 1) 本地知识库检索
        for res in retrieve(q, top_k=2):
            text = res.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                all_results.append(res)
        # 2) 联网搜索（可开关）
        if ENABLE_WEB_SEARCH:
            for res in web_search(q, top_k=2):
                text = res.get("text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(res)
    return {"search_results": all_results}


def code_executor_node(state: ResearchState) -> Dict[str, Any]:
    """
    代码执行节点：结合检索资料判断是否需要运行Python代码做数据分析绘图
    如果需要，让大模型生成Python代码，调用execute_python_code隔离运行，保存输出与图片
    :param state: 全局状态，读取search_results、sub_questions
    :return: 更新 code_outputs、code_images、execution_log
    """
    search_results = state.get("search_results", [])
    # 如果没有检索资料，直接跳过代码执行
    if not search_results:
        return {"code_outputs": [], "code_images": [], "execution_log": "无检索结果，跳过代码执行"}

    # 取前5条检索片段，拼接成给大模型的上下文，每个片段最多截取200字符防止token过多
    context = "\n".join([f"- {res['text'][:200]}..." for res in search_results[:5]])
    sub_qs = state.get("sub_questions", [])

    prompt = f"""你是一位数据分析专家。现有检索结果：
{context}

子问题：{sub_qs}

请判断是否需要写 Python 代码辅助分析（如统计、绘图、计算）。如需，请输出完整 Python 代码（使用 pandas, matplotlib），代码应输出文本描述或保存图表到当前目录。否则回复 "NO_CODE"。
注意：只能使用标准库及 pandas, numpy, matplotlib，禁止导入 os, sys, subprocess 等危险模块。
输出只有代码或 "NO_CODE"。
"""
    response = call_llm(prompt).strip()

    # 大模型回复NO_CODE，代表不需要运行代码
    if response == "NO_CODE":
        return {"code_outputs": [], "code_images": [], "execution_log": "无需代码执行"}

    # 调用隔离执行函数运行AI生成的代码，最长允许运行15秒
    result = execute_python_code(response, timeout_sec=15)
    outputs = []
    # 收集stdout打印输出
    if result["stdout"]:
        outputs.append(f"[输出]\n{result['stdout']}")
    # 收集stderr报错信息
    if result["stderr"]:
        outputs.append(f"[错误]\n{result['stderr']}")
    # 遍历代码生成出来的文件，区分图表图片和普通文件
    if result["generated_files"]:
        for f in result["generated_files"]:
            if f.lower().endswith((".png", ".jpg")):
                outputs.append(f"[图表] {f}")
            else:
                outputs.append(f"[文件] {f}")

    return {
        "code_outputs": outputs,
        "code_images": [f for f in result["generated_files"] if f.lower().endswith((".png", ".jpg"))],
        "execution_log": f"耗时 {result['elapsed']:.2f}s，成功: {result['success']}"
    }


def writer_node(state: ResearchState) -> Dict[str, Any]:
    """
    报告撰写节点：结合检索资料、代码运行输出、上一轮评审修改建议，生成报告草稿
    :param state: 全局状态，读取问题、检索结果、代码输出、评审反馈
    :return: 更新draft字段，保存报告草稿
    """
    question = state["research_question"]
    sub_qs = state.get("sub_questions", [])
    results = state.get("search_results", [])
    code_outputs = state.get("code_outputs", [])
    code_images = state.get("code_images", [])
    feedback = state.get("review_feedback", "")

    # 拼接参考资料文本：给每条编号（[来源1][来源2]...），并带上来源文件名。
    # 用 [来源N] 而非 [N]，是为了和 eval 的引用提取精确对齐，避免正文/代码输出里的普通 [数字] 被误当成引用。
    ref_text = "\n".join(
        f"[来源{idx}] {res['text']} (来源: {res['metadata'].get('source', '未知')})"
        for idx, res in enumerate(results, 1)
    )
    # 拼接代码输出；无输出就填写文字提示
    code_text = "\n".join(code_outputs) if code_outputs else "无代码分析结果"
    img_text = "\n".join(code_images) if code_images else "无图表"

    prompt = f"""请根据以下素材撰写一份研究报告，回答研究问题：{question}
子问题：{sub_qs}
检索内容：
{ref_text}
代码分析结果：
{code_text}
图表：
{img_text}
{ "修改建议：" + feedback if feedback else "" }

报告应结构清晰，包含摘要、各子问题分析、结论。
引用事实或数据时，请在句末用 [来源N] 标注来源编号（如 [来源1]），N 对应上方"检索内容"里的编号。
"""
    draft = call_llm(prompt)
    return {"draft": draft}


def reviewer_node(state: ResearchState) -> Dict[str, Any]:
    """
    评审节点：大模型充当评审专家，给报告打分（0‑10），输出修改建议
    分数≥8分判定合格，或者达到最大迭代次数强制定稿
    :param state: 全局状态，读取草稿、检索原始资料、迭代次数
    :return: 更新review_score、review_feedback、iteration、final_report
    """
    draft = state.get("draft", "")
    # 如果草稿为空，直接判定0分
    if not draft:
        return {
            "review_score": 0.0,
            "review_feedback": "草稿为空",
            "iteration": state.get("iteration", 0) + 1,
            "final_report": None
        }

    # 取出全部原始检索资料，交给评审大模型用来核对报告内容有没有编造信息
    all_texts = "\n".join([res["text"] for res in state.get("search_results", [])])

    prompt = f"""作为评审专家，请评审以下报告草稿：
---报告---
{draft}
---检索原始资料---
{all_texts}

评审标准：
1. 完整性：是否覆盖所有子问题？
2. 引用真实性：文中的引用是否与原始资料匹配？有无编造？
3. 逻辑清晰度：论述是否连贯、有说服力？
4. 是否合理使用了代码分析结果？

请给出评分（0-10分，8分为合格）和具体修改建议。
输出格式：
评分: X分
修改建议: ...
"""
    response = call_llm(prompt)
    # 使用正则表达式提取大模型输出中的数字评分
    score_match = re.search(r"评分:\s*(\d+(?:\.\d+)?)", response)
    score = float(score_match.group(1)) if score_match else 0.0
    feedback = response.strip()

    # 迭代计数自增1
    iteration = state.get("iteration", 0) + 1
    final_report = None
    # 满足条件：分数大于等于7合格，或者达到最大迭代，直接把当前草稿作为最终报告
    if score >= 8 or iteration >= MAX_ITERATIONS:
        final_report = draft

    return {
        "review_score": score,
        "review_feedback": feedback,
        "iteration": iteration,
        "final_report": final_report
    }

# ---------------------- 条件路由配置 ----------------------

def should_continue(state: ResearchState) -> Literal["writer", "end"]:
    """
    条件边路由函数，评审结束后判断下一步去哪里
    :param state:全局状态，只读，不修改任何字段
    :return: 返回字符串"writer"回到写报告重写；返回"end"结束整个工作流
    """
    # 如果已经产出最终报告，直接结束
    if state.get("final_report") is not None:
        return "end"
    # 如果迭代次数达到上限，强制结束流程
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "end"
    # 没有合格报告，且还没有达到最大轮次，回到writer重写草稿
    return "writer"

# ---------------------- 构建LangGraph工作流图 ----------------------
def build_graph():
    """
    组装整个Agent流程图，注册全部节点、连线、条件分支，返回编译完成可运行图对象
    """
    builder = StateGraph(ResearchState)
    # 注册节点：节点名字字符串 + 对应的处理函数
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    # 设置图入口，程序启动第一个运行planner节点
    builder.set_entry_point("planner")

    # 普通固定边，执行完A一定执行B
    builder.add_edge("planner", "researcher")      # 问题拆解完成→检索资料
    builder.add_edge("researcher", "code_executor")# 检索完成→执行代码分析
    builder.add_edge("code_executor", "writer")    # 代码执行完毕→写报告
    builder.add_edge("writer", "reviewer")         # 报告写完→交给评审

    # 条件分支边：评审结束后调用should_continue函数动态选择下一步
    builder.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "writer": "writer",   # 返回writer就跳转到撰写节点重写
            "end": END            # 返回end跳转到LangGraph内置结束标记END
        }
    )
    # compile编译图，得到可以invoke运行的实例
    return builder.compile()

# ---------------------- 程序入口，脚本直接运行时执行 ----------------------
if __name__ == "__main__":
    graph = build_graph()
    print("=" * 50)
    print("AutoResearch 多智能体深度研究系统")
    print("=" * 50)
    print("输入研究问题，系统会自动完成：拆解 → 检索 → 代码执行 → 写作 → 评审。")
    print("（直接回车用示例问题；输入 quit 退出）\n")

    DEMO_QUESTION = "新能源汽车电池回收技术对环境的影响"

    while True:
        question = input("请输入研究问题：").strip()
        if question.lower() == "quit":
            print("再见！")
            break
        if not question:
            question = DEMO_QUESTION
            print(f"（使用示例问题）{question}")

        print("\n正在研究……（通常需 1~3 分钟）\n")
        # 初始化完整初始状态，所有字段赋予初始值
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
            "review_score": None
        }
        # invoke 启动整个 Agent 工作流，运行直到 END，拿到最终状态
        final_state = graph.invoke(initial_state)

        print("\n" + "=" * 50)
        print("最终报告")
        print("=" * 50)
        print(final_state.get("final_report", "未生成报告"))
        print(f"\n重写次数: {final_state.get('iteration', 0)}")
        print(f"评审得分: {final_state.get('review_score', 'N/A')}")
        print("-" * 50 + "\n")