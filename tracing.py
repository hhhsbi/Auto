"""
tracing.py —— LangSmith 监控封装（阶段 2）

统一导出 traceable / wrap_openai / get_current_run_tree，并处理两个降级场景：
  1) langsmith 未安装：导出的 traceable 变成空装饰器，wrap_openai 原样返回客户端，
     系统功能完全不受影响；
  2) 未配置 LANGCHAIN_API_KEY：setup_tracing() 会把 LANGCHAIN_TRACING_V2 置为 false，
     避免每次请求都往 stderr 打 "无法上报" 的警告。

使用方式：
  - main.py 启动时调用一次 setup_tracing()；
  - 各节点函数用 @traceable(name=..., run_type=...) 装饰；
  - call_llm 的 OpenAI 客户端用 wrap_openai 包一层，token 用量自动上报。

LangSmith 面板：https://smith.langchain.com （项目名默认 AutoResearch，
可用 LANGCHAIN_PROJECT 环境变量覆盖）。
"""
import os

try:
    from langsmith import traceable, get_current_run_tree
    from langsmith.wrappers import wrap_openai
    HAS_LANGSMITH = True
except ImportError:  # langsmith 未安装：降级为空实现
    HAS_LANGSMITH = False

    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        # 兼容 @traceable 与 @traceable(...) 两种用法
        return decorator if args and callable(args[0]) else decorator

    def get_current_run_tree():
        return None

    def wrap_openai(client):
        return client


def add_run_metadata(metadata: dict):
    """向当前 trace run 附加自定义元数据（如检索召回条数、评审得分）。
    不在 trace 上下文里或 langsmith 未安装时静默跳过。"""
    try:
        rt = get_current_run_tree()
        if rt is not None:
            rt.metadata = {**(rt.metadata or {}), **metadata}
    except Exception:
        pass  # 元数据上报失败绝不能影响主流程


def setup_tracing() -> bool:
    """
    初始化 LangSmith 追踪开关，返回是否真正开启。

    规则：配置了 LANGCHAIN_API_KEY（在 .env 里）才开启 LANGCHAIN_TRACING_V2，
    否则关闭，避免无 key 时反复打印上报失败警告。
    """
    enabled = bool(os.getenv("LANGCHAIN_API_KEY"))
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    project = os.getenv("LANGCHAIN_PROJECT", "AutoResearch")
    if enabled:
        print(f"[tracing] LangSmith 追踪已开启，项目: {project}")
    else:
        print("[tracing] LangSmith 追踪未开启（在 .env 配置 LANGCHAIN_API_KEY 后自动启用）")
    return enabled
