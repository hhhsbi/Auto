# AutoResearch —— 基于 LangGraph 的多智能体深度研究系统

AutoResearch 是一个多智能体深度研究系统：输入一个研究问题，系统自动拆解为子问题，
由**规划 / 检索 / 代码执行 / 写作 / 评审**多个角色协作完成研究，经过反思循环迭代修正，
最终输出**带引用来源、含数据分析图表**的结构化报告。

## 架构

```
                         ┌─────────────────────────────┐
                         │  用户输入一个研究问题          │
                         └──────────────┬──────────────┘
                                        ▼
                            ┌─────────────────────┐
                            │  ① Planner（规划）    │  拆解为 3~5 个子问题
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  ② Researcher（检索） │  向量库 RAG 检索相关片段
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  ③ Code Executor    │  沙箱跑 Python 算数据/画图
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  ④ Writer（写作）    │  生成带引用的报告草稿
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  ⑤ Reviewer（评审）  │  打分 + 引用真实性检查
                            └──────────┬──────────┘
                                       │ 不合格 & 未达上限 → 打回 Writer 重写
                                       ▼
                                Final Report（最终报告）
```

四角色核心是「规划 / 检索 / 写作 / 评审」，代码执行作为检索与写作之间的增强环节。

## 功能特性

- **自动问题拆解**：Planner 把大问题拆成可检索的子问题。
- **向量检索（RAG）**：基于 Chroma + 多语言 Sentence-BERT，按语义检索知识库。
- **安全代码执行**：子进程 + 临时目录 + 超时控制，隔离运行 LLM 生成的 Python（pandas/matplotlib）。
- **反思循环**：Reviewer 给报告打分并检查引用真实性，不合格则打回 Writer 重写，最多 3 轮。
- **带引用输出**：报告事实标注来源编号，可追溯。
- **可量化评估**：内置 10 道跨领域测试题 + 引用准确率/答案质量评估脚本。

## 技术栈

- **语言**：Python
- **LLM**：DeepSeek（OpenAI 兼容接口，`deepseek-chat`）
- **编排**：LangGraph（StateGraph 状态图 + 条件边）
- **向量库**：Chroma（持久化）
- **Embedding**：sentence-transformers（`paraphrase-multilingual-MiniLM-L12-v2`）
- **数据分析**：pandas / numpy / matplotlib（代码执行沙箱内）

## 目录结构

```
agent_AutoResearch/
├── main.py              # 主流水线：规划/检索/代码执行/写作/评审 + 反思循环
├── rag.py               # RAG 检索模块（retrieve() / index_documents()）
├── eval/                # 评估：测试题 + 评估脚本 + 对比模板
│   ├── test_questions.py
│   ├── evaluate.py
│   └── results_template.md
├── docs/                # 知识库源文件（.txt / .md）
├── chroma_data/         # Chroma 持久化存储（运行时生成）
├── outputs/             # 代码执行生成的图表（运行时生成）
└── requirements.txt     # 依赖清单
```

## 快速开始

```bash
# 0. 所有命令请在项目根目录（agent_AutoResearch/）运行，否则相对路径 ./docs、./chroma_data 会找不到

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥（.env 文件，已在 .gitignore 中，不要上传到 Git）
echo "DEEPSEEK_API_KEY=你的密钥" > .env

# 3. 运行系统（完整流水线）
python main.py

# 4. 运行评估
python eval/evaluate.py
```

> 首次运行会从镜像 `hf-mirror.com` 下载 embedding 模型（已在代码中设置 `HF_ENDPOINT`）。
> 联网搜索默认关闭（国内网络下 DuckDuckGo 被墙）。若网络可用，设环境变量 `ENABLE_WEB_SEARCH=true` 即可开启「本地知识库 + 联网」双路检索。

## 评估结果

在 10 道跨领域测试题（知识库内 6 题 / 代码执行 2 题 / 知识库外 2 题）上，
以「引用准确率」和「答案质量」两个指标评估：

| 指标 | 优化前（无反思） | 优化后（有反思） |
|------|:---:|:---:|
| 平均引用准确率 | ___% | ___% |
| 平均答案质量（0-100） | ___ | ___ |

> 填上 `python eval/evaluate.py` 跑出的数字即可。
