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
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prebuilt import ToolExecutor
from duckduckgo_search import DDGS  # 联网搜索（DuckDuckGo）

# 导入阶段2的真实 RAG 检索（本地知识库），替代原来的 mock retrieve
from rag import retrieve, index_documents
# LangSmith 监控（阶段 2）：langsmith 未安装时这些导出自动降级为空实现
from tracing import traceable, wrap_openai, add_run_metadata, setup_tracing

# 项目根目录（绝对路径），保证从任何工作目录运行都能正确定位 docs/outputs。
BASE_DIR = Path(__file__).resolve().parent

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set")

# 初始化 LangSmith 追踪开关（配置了 LANGCHAIN_API_KEY 才真正开启）
setup_tracing()

# 联网搜索开关：默认关闭（国内网络下 DuckDuckGo 被墙，开启会一直失败并拖慢评估）。
# 网络可用时设环境变量 ENABLE_WEB_SEARCH=true，即可开启"本地知识库 + 联网"双路检索。
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"

# 反思循环最大迭代轮次（防止无限循环；评估时设为 1 即为"无反思"基线）
MAX_ITERATIONS = 3
# 阶段5 Step 6：动态规划迭代上限，防止 reflection_planner 无限追加子问题
MAX_PLANNER_ITERATIONS = 2

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
    # ========== 新增：RBAC 用户上下文（阶段3）==========
    # 由 /ask 从 JWT 解出后传入，researcher 据此用 Chroma where 过滤检索范围
    user_role: Optional[str]  # 当前用户角色；None=未认证，只能检索公开文档
    user_department: Optional[str]  # 当前用户部门，配合 allowed_departments 标签过滤
    # ========== 新增：多轮对话上下文（阶段5 Step 5）==========
    # chat_history: 之前的对话历史，格式 [{role:"user",content:"..."},{role:"assistant",content:"..."}]
    # writer_node 会把历史拼进 prompt，让 LLM 知道上文，理解"那它呢"这种追问
    chat_history: List[Dict[str, str]]
    session_id: Optional[str]  # 会话 ID；None=单轮（保持兼容），有值时问答会追加到 Redis 历史
    # ========== 新增：动态规划（阶段5 Step 6）==========
    # planner_iteration: reflection_planner 已追加子问题的次数；超过 MAX_PLANNER_ITERATIONS 强制走 code_executor
    planner_iteration: int
    # need_replan: reflection_planner 设置的路由标记——True=追加了子问题需回 researcher 重检索
    need_replan: bool
    # ========== 新增：SSE 流式 + HITL（阶段5 Step 7）==========
    # awaiting_user: planner 后是否暂停等待用户确认/编辑子问题（HITL 检查点）
    awaiting_user: bool
    # user_edit: 用户通过 /resume 提交的编辑后子问题（JSON 字符串），planner_node 会解析使用
    user_edit: Optional[str]
    # ========== 新增：意图路由（闲聊 vs 研究）==========
    # intent: router_node 判定的意图——"chat"=闲聊直接回，"research"=走完整研究流水线
    intent: str


# LLM 客户端模块级复用（原来每次调用新建一个）；
# wrap_openai 让每次 LLM 调用的 token 用量自动上报到 LangSmith
_llm_client = wrap_openai(openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,              # 修复：原来误写成字符串 "DEEPSEEK_API_KEY"
    base_url="https://api.deepseek.com",   # 修复：原来误写成 deepseed.com
))


@traceable(name="call_llm", run_type="llm")
def call_llm(prompt: str) -> str:
    response = _llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],  # 修复：原来误写成 "context"
        temperature=0.3,
    )
    return response.choices[0].message.content


@traceable(name="execute_python_code", run_type="tool")
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


@traceable(name="web_search", run_type="retriever")
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


# ────────────────────────── 意图路由：闲聊 vs 研究 ──────────────────────────
# 闲聊匹配模式：问候、致谢、确认、简短寒暄——不需要走研究流水线
import re as _re
_CHAT_PATTERNS = _re.compile(
    r'^(你好|您好|hi|hello|hey|嗨|在吗|在不在|谢谢|感谢|辛苦了|好的|嗯|ok|bye|再见|'
    r'你是谁|你叫什么|你能做什么|帮我什么|介绍下你自己|'
    r'哈喽|哈罗|早上?好|下午好|晚上好|晚安).*$',
    _re.IGNORECASE
)


@traceable(name="router", run_type="chain")
def router_node(state: ResearchState) -> Dict[str, Any]:
    """
    意图路由节点：判断用户问题是闲聊还是研究。

    策略（先规则后 LLM，省 token）：
      1) 匹配闲聊模式（你好/谢谢/你是谁等）→ "chat"
      2) 问题很短（<8 字）且不含问号/研究关键词 → "chat"
      3) 有历史且是简短追问（如"继续""详细说说"）→ "chat"（基于历史直接回）
      4) 其它 → "research"（走完整流水线）
    """
    question = state["research_question"].strip()
    history = state.get("chat_history", [])

    # 1) 匹配闲聊模式
    if _CHAT_PATTERNS.match(question):
        add_run_metadata({"意图": "闲聊（模式匹配）"})
        return {"intent": "chat"}

    # 2) 有问号 → 研究意图（带问号的问题大概率需要回答/分析）
    if "?" in question or "？" in question:
        add_run_metadata({"意图": "研究（含问号）"})
        return {"intent": "research"}

    # 3) 包含研究关键词 → 研究意图
    research_kws = ["分析", "研究", "调研", "报告", "总结", "对比", "原理",
                    "什么是", "解释", "介绍", "阐述", "说明", "评估", "方案",
                    "影响", "趋势", "发展", "应用", "区别", "优缺", "实现",
                    "如何", "怎样", "为什么", "为何"]
    if any(kw in question for kw in research_kws):
        add_run_metadata({"意图": "研究（关键词命中）"})
        return {"intent": "research"}

    # 4) 有历史 + 简短追问 → 基于历史直接回答
    if history and len(question) < 15:
        follow_ups = ["继续", "详细说说", "展开", "然后呢", "那它呢", "接着说",
                      "多说点", "举个例子", "具体点", "深入点"]
        if any(q in question for q in follow_ups):
            add_run_metadata({"意图": "闲聊（追问）"})
            return {"intent": "chat"}

    # 5) 极短且不含研究关键词 → 闲聊
    if len(question) < 8:
        add_run_metadata({"意图": "闲聊（短消息）"})
        return {"intent": "chat"}

    # 6) 默认走研究流水线
    add_run_metadata({"意图": "研究"})
    return {"intent": "research"}


def should_research(state: ResearchState) -> Literal["chat", "planner"]:
    """条件边：router 之后，intent=chat → chat_node，intent=research → planner"""
    if state.get("intent") == "chat":
        return "chat"
    return "planner"


@traceable(name="chat", run_type="chain")
def chat_node(state: ResearchState) -> Dict[str, Any]:
    """
    闲聊节点：直接调 LLM 回复，不走研究流水线。

    拼入对话历史让 LLM 有上下文（如"继续"知道继续什么）。
    返回 final_report 让 /ask 和 SSE 的后续逻辑无缝衔接。
    """
    question = state["research_question"]
    history = state.get("chat_history", [])

    # 构造聊天 prompt（含历史，让追问有上下文）
    msgs = [{"role": "system", "content":
        "你是 AutoResearch 智能研究助手。用户正在和你聊天。"
        "简洁友好地回复，一两句话即可。"
        "如果用户的问题需要深入分析、检索资料、写报告，请提示用户可以直接提问，你会自动展开研究。"}]
    for h in history[-6:]:  # 最近 3 轮（6 条）
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": question})

    response = _llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=msgs,
        temperature=0.7,  # 闲聊可以稍微多样
        max_tokens=500,   # 闲聊不需要长回复
    )
    reply = response.choices[0].message.content
    return {"final_report": reply}


@traceable(name="planner", run_type="chain")
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


@traceable(name="researcher", run_type="chain")
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
    # 用户权限上下文：传给 retrieve 做 Chroma where 过滤，不同角色检索范围不同
    user_role = state.get("user_role")
    user_department = state.get("user_department")
    all_results = []
    seen_texts = set()  # 按正文去重，避免本地和联网结果重复
    for q in sub_qs:
        # 1) 本地知识库检索（带权限过滤：只返回该用户有权看的块）
        for res in retrieve(q, top_k=2, role=user_role, department=user_department):
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
    # 附加监控元数据：召回条数 / 启用的检索通道 / 用户权限上下文（LangSmith 面板可见）
    add_run_metadata({
        "检索召回条数": len(all_results),
        "子问题数": len(sub_qs),
        "联网检索": ENABLE_WEB_SEARCH,
        "用户角色": user_role or "(未认证)",
        "用户部门": user_department or "(未认证)",
    })
    return {"search_results": all_results}


# ────────────────────────── 阶段5 Step 6：动态规划（reflection planner）──────────────────────────
@traceable(name="reflection_planner", run_type="chain")
def reflection_planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    反思规划节点：分析检索结果是否充分，不充分时追加子问题并标记回 researcher 重检索。

    轻量 ReAct 策略（避免过度调用 LLM）：
      1) 已达 MAX_PLANNER_ITERATIONS 上限 → 不再规划，直接放行
      2) 检索结果太少（< 2 条）→ 让 LLM 补充 1-2 个不同角度子问题
      3) 结果较多时 → 让 LLM 判断覆盖是否充分，不充分则补 1 个子问题
      4) 结果充分 → 放行

    通过 need_replan 布尔标记控制 should_replan 路由：
      need_replan=True  → 回 researcher 用新子问题重检索
      need_replan=False → 进 code_executor 继续主流程
    """
    planner_iter = state.get("planner_iteration", 0)

    # 1) 已达迭代上限，不再追加子问题
    if planner_iter >= MAX_PLANNER_ITERATIONS:
        add_run_metadata({"动态规划": f"已达上限({MAX_PLANNER_ITERATIONS})，放行"})
        return {"need_replan": False}

    results = state.get("search_results", [])
    question = state["research_question"]
    sub_qs = state.get("sub_questions", [])

    # 2) 检索结果太少，直接让 LLM 补充子问题
    if len(results) < 2:
        prompt = f"""研究问题：{question}
已有子问题：{sub_qs}
检索到的资料极少（{len(results)}条），难以回答研究问题。
请基于已有子问题，补充 1-2 个不同角度的子问题，帮助找到更多相关资料。
只输出补充的子问题，每行一个，不要序号。"""
        response = call_llm(prompt)
        new_qs = [line.strip() for line in response.split("\n") if line.strip()
                  and not line.strip().startswith("研究问题") and not line.strip().startswith("已有子问题")]
        if new_qs:
            add_run_metadata({"动态规划": f"追加{len(new_qs)}个子问题（结果少）", "新增子问题": new_qs})
            return {
                "sub_questions": sub_qs + new_qs,
                "planner_iteration": planner_iter + 1,
                "need_replan": True,
            }
        add_run_metadata({"动态规划": "LLM 未返回子问题，放行"})
        return {"need_replan": False}

    # 3) 结果较多时，让 LLM 判断覆盖是否充分
    ref_summary = "\n".join(f"- {r['text'][:150]}" for r in results[:5])
    prompt = f"""研究问题：{question}
已有子问题：{sub_qs}
检索到的资料摘要：
{ref_summary}

请判断这些资料是否足以回答研究问题。如果某个关键方面缺少资料覆盖，请补充 1 个子问题。
如果资料已充分覆盖，请只回复 "SUFFICIENT"。
只输出补充的子问题（或 SUFFICIENT），不要序号，不要解释。"""
    response = call_llm(prompt).strip()
    if response and response != "SUFFICIENT":
        new_qs = [line.strip() for line in response.split("\n") if line.strip()]
        if new_qs:
            add_run_metadata({"动态规划": f"追加{len(new_qs)}个子问题（覆盖不足）", "新增子问题": new_qs})
            return {
                "sub_questions": sub_qs + new_qs,
                "planner_iteration": planner_iter + 1,
                "need_replan": True,
            }

    # 4) 结果充分，放行
    add_run_metadata({"动态规划": "资料充分，放行"})
    return {"need_replan": False}


def should_replan(state: ResearchState) -> Literal["researcher", "code_executor"]:
    """
    条件边：reflection_planner 之后判断下一步去向。
    need_replan=True → 回 researcher 重检索（用新追加的子问题）
    need_replan=False → 进 code_executor 继续主流程
    """
    if state.get("need_replan", False):
        return "researcher"
    return "code_executor"


@traceable(name="code_executor", run_type="chain")
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

    add_run_metadata({
        "代码执行成功": result["success"],
        "代码执行耗时s": round(result["elapsed"], 2),
    })
    return {
        "code_outputs": outputs,
        "code_images": [f for f in result["generated_files"] if f.lower().endswith((".png", ".jpg"))],
        "execution_log": f"耗时 {result['elapsed']:.2f}s，成功: {result['success']}"
    }


@traceable(name="writer", run_type="chain")
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

    # 阶段5 Step 5：多轮对话上下文——把历史拼进 prompt，让 LLM 知道上文
    # 按简单 token 预算截断：最多保留最近 6 轮（12 条），避免历史超长撑爆 prompt
    chat_history = state.get("chat_history", []) or []
    if chat_history:
        recent = chat_history[-12:]  # 最近最多 12 条
        history_text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '研究助手'}：{m.get('content', '')[:500]}"
            for m in recent
        )
        history_block = f"""
【对话历史（如本次问题是追问，请结合历史理解上下文，但报告仍聚焦回答本次问题）】
{history_text}
"""
    else:
        history_block = ""

    prompt = f"""请根据以下素材撰写一份研究报告，回答研究问题：{question}
{history_block}
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


@traceable(name="reviewer", run_type="chain")
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
    # 满足条件：分数大于等于8合格，或者达到最大迭代，直接把当前草稿作为最终报告
    if score >= 8 or iteration >= MAX_ITERATIONS:
        final_report = draft

    # 评审结果写入 LangSmith 元数据，便于面板按分数/轮次过滤
    add_run_metadata({"评审得分": score, "迭代轮次": iteration})
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
def build_graph(checkpointer=None):
    """
    组装整个Agent流程图，注册全部节点、连线、条件分支，返回编译完成可运行图对象。

    阶段5 Step 6 新增 reflection_planner 节点（在 researcher 和 code_executor 之间）：
      planner → researcher → reflection_planner →(should_replan)→ researcher（重检索）或 code_executor
      → writer → reviewer →(should_continue)→ writer（重写）或 END

    阶段5 Step 7：checkpointer 参数用于流式 HITL（interrupt + resume）；
    不传时编译普通图（/ask 同步用），传 MemorySaver 时编译带 checkpoint 的流式图。
    """
    builder = StateGraph(ResearchState)
    # 注册节点：节点名字字符串 + 对应的处理函数
    builder.add_node("router", router_node)             # 意图路由：闲聊 vs 研究
    builder.add_node("chat", chat_node)                 # 闲聊直接回 LLM
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("reflection_planner", reflection_planner_node)  # Step 6：动态规划节点
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    # 设置图入口：先走 router 判意图
    builder.set_entry_point("router")

    # 普通固定边
    builder.add_edge("chat", END)                        # 闲聊直接结束
    builder.add_edge("planner", "researcher")           # 问题拆解完成→检索资料
    builder.add_edge("researcher", "reflection_planner") # 检索完成→反思规划（Step 6）
    builder.add_edge("code_executor", "writer")          # 代码执行完毕→写报告
    builder.add_edge("writer", "reviewer")               # 报告写完→交给评审

    # 意图路由条件边：router 之后判断走 chat 还是 planner
    builder.add_conditional_edges(
        "router",
        should_research,
        {
            "chat": "chat",          # 闲聊→chat_node 直接回
            "planner": "planner",    # 研究→planner 走完整流水线
        }
    )

    # Step 6 条件边：reflection_planner 之后判断是否需要重检索
    builder.add_conditional_edges(
        "reflection_planner",
        should_replan,
        {
            "researcher": "researcher",     # 需要追加子问题→回 researcher 重检索
            "code_executor": "code_executor"  # 资料充分→继续代码执行
        }
    )

    # 条件分支边：评审结束后调用should_continue函数动态选择下一步
    builder.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "writer": "writer",   # 返回writer就跳转到撰写节点重写
            "end": END            # 返回end跳转到LangGraph内置结束标记END
        }
    )
    # compile编译图，得到可以invoke运行的实例（Step 7：checkpointer 为流式 HITL 用）
    return builder.compile(checkpointer=checkpointer)


# 编译后的图缓存：run_research 每次调用复用同一实例
_graph = None


@traceable(name="AutoResearch 研究流水线", run_type="chain")
def run_research(question: str, role: Optional[str] = None, department: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None,
                 session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    跑一遍完整研究流水线的对外入口（LangSmith 根 trace）。

    LangGraph 的节点在同一调用栈里同步执行，所以 planner/researcher/
    code_executor/writer/reviewer 各节点的 @traceable 运行记录会自动
    挂到这个根 run 下，在 LangSmith 里形成完整链路树。

    :param question: 用户研究问题
    :param role: 当前用户角色（来自 JWT），控制检索范围；None=未认证只看公开文档
    :param department: 当前用户部门（来自 JWT），控制检索范围
    :param history: 多轮对话历史（阶段5 Step 5），格式 [{role,content},...]
                    None 或空列表=单轮（保持兼容）；非空时 writer_node 会拼进 prompt
    :param session_id: 会话 ID；None=单轮；有值时由 app.py 负责把问答追加到 Redis 历史
    :return: 流程结束后的最终状态字典
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    add_run_metadata({"用户": role and f"{role}@{department}" or "(未认证)"})
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
        "user_role": role,
        "user_department": department,
        # 阶段5 Step 5：多轮对话上下文
        "chat_history": history or [],
        "session_id": session_id,
        # 阶段5 Step 6：动态规划
        "planner_iteration": 0,
        "need_replan": False,
        # 阶段5 Step 7：SSE 流式 + HITL
        "awaiting_user": False,
        "user_edit": None,
        # 意图路由
        "intent": "",
    }
    return _graph.invoke(initial_state)


# ────────────────────────── 阶段5 Step 7：SSE 流式 + HITL ──────────────────────────

# 流式图专用编译实例（带 MemorySaver checkpointer，支持 interrupt + resume）
_streaming_graph = None
# checkpointer 实例（跨请求共享，/ask_stream 写入、/resume 读取）
_saver = MemorySaver()


def _get_streaming_graph():
    """获取带 checkpointer 的流式图实例（懒加载，首次调用时编译）。"""
    global _streaming_graph
    if _streaming_graph is None:
        _streaming_graph = build_graph(checkpointer=_saver)
    return _streaming_graph


def _build_initial_state(question: str, role: Optional[str], department: Optional[str],
                         history: Optional[List[Dict[str, str]]],
                         session_id: Optional[str]) -> dict:
    """构造初始状态（run_research 和 run_research_stream 共用）。"""
    return {
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
        "user_role": role,
        "user_department": department,
        "chat_history": history or [],
        "session_id": session_id,
        "planner_iteration": 0,
        "need_replan": False,
        "awaiting_user": False,
        "user_edit": None,
        "intent": "",
    }


# 节点中文名映射（SSE 事件里用中文标签，前端直接显示）
_NODE_LABELS = {
    "router": "意图判断",
    "chat": "闲聊回复",
    "planner": "规划拆解",
    "researcher": "检索资料",
    "reflection_planner": "反思规划",
    "code_executor": "代码分析",
    "writer": "撰写报告",
    "reviewer": "评审报告",
}


def _format_node_event(node_name: str, state_update: dict) -> dict:
    """把 LangGraph 的节点更新格式化为前端可用的 SSE 事件 payload。"""
    label = _NODE_LABELS.get(node_name, node_name)
    payload = {"type": "node", "node": node_name, "label": label}

    # 按节点提取关键信息（不泄露全部内部状态）
    if node_name == "router":
        payload["intent"] = state_update.get("intent", "research")
    elif node_name == "chat":
        # 闲聊节点直接产出 final_report，前端可以直接显示
        final = state_update.get("final_report")
        if final:
            payload["final_report"] = final
    elif node_name == "planner":
        payload["sub_questions"] = state_update.get("sub_questions", [])
    elif node_name == "researcher":
        results = state_update.get("search_results", [])
        payload["search_count"] = len(results)
        payload["sources"] = [r.get("metadata", {}).get("source", "") for r in results[:5]]
    elif node_name == "reflection_planner":
        payload["planner_iteration"] = state_update.get("planner_iteration", 0)
        payload["need_replan"] = state_update.get("need_replan", False)
        if state_update.get("sub_questions"):
            payload["sub_questions"] = state_update["sub_questions"]
    elif node_name == "code_executor":
        payload["execution_log"] = state_update.get("execution_log", "")
        payload["has_images"] = len(state_update.get("code_images", [])) > 0
    elif node_name == "writer":
        draft = state_update.get("draft", "")
        payload["draft_preview"] = draft[:200] if draft else ""
    elif node_name == "reviewer":
        payload["review_score"] = state_update.get("review_score")
        payload["iteration"] = state_update.get("iteration", 0)
        final = state_update.get("final_report")
        if final:
            payload["final_report"] = final

    return payload


def run_research_stream(question: str, role: Optional[str] = None,
                        department: Optional[str] = None,
                        history: Optional[List[Dict[str, str]]] = None,
                        session_id: Optional[str] = None,
                        hitl: bool = False,
                        thread_id: Optional[str] = None):
    """
    流式版 run_research：yield SSE 格式的事件，前端可实时看到每个节点进度。

    闲聊意图：直接调 LLM stream=True，逐 token 推送到前端（秒级首字）。
    研究意图：走 LangGraph 节点流式，每个节点完成后推送进度事件。

    :param hitl: True=planner 后暂停等待用户确认子问题（HITL 检查点）
    :param thread_id: 执行线程 ID；HITL 模式下 /resume 用同一 thread_id 恢复
    :yield: SSE 字符串 "data: {...}\\n\\n"
    """
    import json as _json

    tid = thread_id or f"run-{int(time.time() * 1000)}"

    # 发送 start 事件
    start_event = _json.dumps({
        "type": "start", "thread_id": tid, "question": question,
        "hitl": hitl,
    }, ensure_ascii=False)
    yield f"data: {start_event}\n\n"

    # ── 意图判断（复用 router 逻辑，不进图，直接判）──
    fake_state = {
        "research_question": question,
        "chat_history": history or [],
    }
    intent = router_node(fake_state).get("intent", "research")
    intent_event = _json.dumps({"type": "node", "node": "router", "label": "意图判断",
                                 "intent": intent}, ensure_ascii=False)
    yield f"data: {intent_event}\n\n"

    # ── 闲聊：直接流式调 LLM，逐 token 推送 ──
    if intent == "chat":
        chat_event = _json.dumps({"type": "node", "node": "chat", "label": "闲聊回复"},
                                  ensure_ascii=False)
        yield f"data: {chat_event}\n\n"

        msgs = [{"role": "system", "content":
            "你是 AutoResearch 智能研究助手。用户正在和你聊天。"
            "简洁友好地回复，一两句话即可。"
            "如果用户的问题需要深入分析、检索资料、写报告，请提示用户可以直接提问，你会自动展开研究。"}]
        for h in (history or [])[-6:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": question})

        full_reply = ""
        stream = _llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=msgs,
            temperature=0.7,
            max_tokens=500,
            stream=True,  # 流式输出
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_reply += delta
                token_event = _json.dumps({"type": "token", "content": delta},
                                           ensure_ascii=False)
                yield f"data: {token_event}\n\n"

        done_event = _json.dumps({
            "type": "done", "thread_id": tid, "final_report": full_reply,
        }, ensure_ascii=False)
        yield f"data: {done_event}\n\n"
        return

    # ── 研究：走 LangGraph 节点流式 ──
    graph = _get_streaming_graph()
    config = {"configurable": {"thread_id": tid}}
    initial_state = _build_initial_state(question, role, department, history, session_id)

    # HITL 模式：planner 后暂停
    interrupt_nodes = ["planner"] if hitl else None

    for event in graph.stream(initial_state, config, stream_mode="updates",
                              interrupt_after=interrupt_nodes):
        if "__interrupt__" in event:
            state = graph.get_state(config)
            sub_qs = state.values.get("sub_questions", [])
            await_event = _json.dumps({
                "type": "awaiting_user",
                "thread_id": tid,
                "sub_questions": sub_qs,
                "message": "请确认或编辑子问题，然后调 /ask_stream/resume 提交",
            }, ensure_ascii=False)
            yield f"data: {await_event}\n\n"
            return

        for node_name, state_update in event.items():
            if node_name.startswith("__"):
                continue
            # 跳过 router/chat 事件（已经在外面处理了）
            if node_name in ("router", "chat"):
                continue
            node_event = _format_node_event(node_name, state_update)
            yield f"data: {_json.dumps(node_event, ensure_ascii=False)}\n\n"

    # 执行完成，发 done 事件
    final_state = graph.get_state(config)
    final_report = final_state.values.get("final_report") or final_state.values.get("draft", "")
    done_event = _json.dumps({
        "type": "done",
        "thread_id": tid,
        "final_report": final_report,
        "iteration": final_state.values.get("iteration", 0),
        "review_score": final_state.values.get("review_score"),
    }, ensure_ascii=False)
    yield f"data: {done_event}\n\n"


def resume_research_stream(thread_id: str, user_sub_questions: List[str]):
    """
    HITL 恢复：用户编辑/确认子问题后，更新 state 并继续流式执行。

    :param thread_id: /ask_stream 返回的 thread_id
    :param user_sub_questions: 用户编辑后的子问题列表
    :yield: SSE 字符串
    """
    import json as _json

    graph = _get_streaming_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # 注入用户编辑的子问题
    graph.update_state(config, {"sub_questions": user_sub_questions, "awaiting_user": False})

    resume_event = _json.dumps({
        "type": "resumed", "thread_id": thread_id,
        "sub_questions": user_sub_questions,
    }, ensure_ascii=False)
    yield f"data: {resume_event}\n\n"

    # 继续流式执行（从中断点恢复，不需要再传 initial_state）
    for event in graph.stream(None, config, stream_mode="updates"):
        if "__interrupt__" in event:
            continue
        for node_name, state_update in event.items():
            if node_name.startswith("__"):
                continue
            node_event = _format_node_event(node_name, state_update)
            yield f"data: {_json.dumps(node_event, ensure_ascii=False)}\n\n"

    # 执行完成
    final_state = graph.get_state(config)
    final_report = final_state.values.get("final_report") or final_state.values.get("draft", "")
    done_event = _json.dumps({
        "type": "done",
        "thread_id": thread_id,
        "final_report": final_report,
        "iteration": final_state.values.get("iteration", 0),
        "review_score": final_state.values.get("review_score"),
    }, ensure_ascii=False)
    yield f"data: {done_event}\n\n"


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
        # 初始化完整初始状态，所有字段赋予初始值（命令行模式无登录上下文，只检索公开文档）
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
            "user_role": None,
            "user_department": None,
            # 阶段5 Step 5/6
            "chat_history": [],
            "session_id": None,
            "planner_iteration": 0,
            "need_replan": False,
            "awaiting_user": False,
            "user_edit": None,
            # 意图路由
            "intent": "",
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