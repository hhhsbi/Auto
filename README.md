# AutoResearch —— 基于 LangGraph 的多智能体深度研究系统

AutoResearch 是一个多智能体深度研究系统：输入一个研究问题，系统自动拆解为子问题，
由**规划 / 检索 / 代码执行 / 写作 / 评审**多个角色协作完成研究，经过反思循环迭代修正，
最终输出**带引用来源、含数据分析图表**的结构化报告。

项目从一个命令行脚本，分 5 个阶段演进出**生产级 RAG 服务**：
HTTP 接口（FastAPI）→ 文件异步入库 + 切块策略 → LangSmith 监控 → RBAC 权限 → 二级缓存与失效。
完整改造记录见 [CHANGES.md](CHANGES.md)，改造指令与知识点见 [生产级RAG改造指令集与知识点.md](生产级RAG改造指令集与知识点.md)。

## 架构

### LangGraph 流水线（核心）

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

### 生产级 RAG 服务层（5 阶段演进）

```
┌──────────────────────────────────────────────────────────────┐
│  FastAPI（app.py）                                            │
│  ┌────────┬────────┬────────┬────────┬────────┬──────────┐  │
│  │/login  │/health │/ask    │/search │/upload │/tasks/:id│  │
│  └────┬───┴────────┴───┬───┴────┬───┴────┬───┴──────────┘  │
│       │ JWT 鉴权(auth.py)│        │ admin  │                 │
│       ▼                ▼        ▼        ▼                 │
│   ┌──────────────────────────────────────────────┐          │
│   │  LangGraph 流水线（main.py: run_research）   │          │
│   │  ↓ 用户 role/dept 注入 ResearchState         │          │
│   └──────────────────────┬───────────────────────┘          │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────┐          │
│   │  RAG 检索（rag.py: retrieve）                │          │
│   │  ↓ Chroma where 权限过滤（先筛再排）          │          │
│   └──────┬───────────────────────────────────────┘          │
│          ▼                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│   │ 一级 LRU 缓存 │  │ 二级 Redis   │  │ LangSmith   │     │
│   │ (cache.py)   │  │ (cache.py)   │  │ (tracing.py)│     │
│   └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│   异步入库（ingest.py: BackgroundTasks + SQLite）            │
└──────────────────────────────────────────────────────────────┘
```

## 功能特性

### 流水线核心
- **自动问题拆解**：Planner 把大问题拆成可检索的子问题。
- **向量检索（RAG）**：基于 Chroma + 多语言 Sentence-BERT，按语义检索知识库。
- **安全代码执行**：子进程 + 临时目录 + 超时控制，隔离运行 LLM 生成的 Python（pandas/matplotlib）。
- **反思循环**：Reviewer 给报告打分并检查引用真实性，不合格则打回 Writer 重写，最多 3 轮。
- **带引用输出**：报告事实标注来源编号，可追溯。
- **可量化评估**：内置 10 道跨领域测试题 + 引用准确率/答案质量评估脚本。

### 生产级能力（5 阶段演进产出）
- **HTTP 服务（阶段 0）**：FastAPI + pydantic 模型校验；`/ask` `/upload` `/health` `/login` `/search` `/tasks` `/cache/*` 接口齐备。
- **文件异步入库（阶段 1）**：`/upload` 收到文件立即返 202 + task_id，后台 BackgroundTasks 完成「解析→切块→向量化→入库」；固定长度滑动窗口切块（500 字 + 50 字重叠，尾部回退到段落/句号边界）；`/tasks/{id}` 查进度；SQLite 元信息表；doc_id 用内容 sha256 实现幂等。
- **LangSmith 监控（阶段 2）**：全节点 `@traceable`；`call_llm` 的 token 用量自动上报；关键节点采集检索召回/执行耗时/评审得分/用户角色；两层优雅降级（未装包/未配 key 不影响业务）。
- **RBAC 权限控制（阶段 3）**：JWT 鉴权；文档入库打 `allowed_roles`/`allowed_departments` 列表标签；Chroma `where` 先筛再排（top_k 只在有权看范围内算）；不同角色同题检索结果不同；上传仅 admin。
- **二级缓存与失效（阶段 4）**：一级 `cachetools.TTLCache` + 二级 Redis（未启降级）；key 含 role/model_version/top_k（权限/模型变更自然失效）；**反向索引**实现按 doc_id 精确失效；`DELETE /cache/clear` 手动清；TTL 5 分钟兜底。

## 技术栈

- **语言**：Python 3.10+
- **LLM**：DeepSeek（OpenAI 兼容接口，`deepseek-chat`）
- **编排**：LangGraph（StateGraph 状态图 + 条件边）
- **向量库**：Chroma（持久化）
- **Embedding**：sentence-transformers（`paraphrase-multilingual-MiniLM-L12-v2`）
- **Web 框架**：FastAPI + uvicorn + pydantic
- **认证**：JWT（pyjwt，HS256）
- **缓存**：Redis（redis-py，protocol=2 兼容 5.x）+ cachetools.TTLCache
- **监控**：langsmith（@traceable + wrap_openai，未配 key 优雅降级）
- **元信息**：SQLite（stdlib sqlite3，documents/tasks 两表）
- **数据分析**：pandas / numpy / matplotlib（代码执行沙箱内）

## 目录结构

```
agent_AutoResearch/
├── main.py              # LangGraph 主流水线：五节点 + 反思循环 + run_research 根 trace
├── rag.py               # RAG 检索：滑动窗口切块 + Chroma 检索 + 权限 where 过滤
├── app.py               # FastAPI 服务层：/login /health /ask /search /upload /tasks /cache
├── auth.py              # JWT 签发/校验 + 三个演示用户 + get_current_user/require_admin
├── ingest.py            # 文件异步入库管道：SQLite 元信息 + 后台切块向量化
├── tracing.py           # LangSmith 监控封装：@traceable/wrap_openai 统一导出 + 优雅降级
├── cache.py             # 两级缓存管理器：LRU + Redis + 反向索引精确失效
├── eval/                # 评估：10 道测试题 + 评估脚本 + 对比模板
│   ├── test_questions.py
│   ├── evaluate.py
│   └── results_template.md
├── docs/                # 知识库源文件（.txt / .md / 含 hr_salary / tech_roadmap 机密样本）
├── chroma_data/         # Chroma 持久化存储（运行时生成）
├── outputs/             # 代码执行生成的图表（运行时生成）
├── rbac_e2e_test.py     # 阶段3验收：RBAC 检索层 17 项断言（持续回归资产）
├── verify_alice_ask.py  # 阶段3验收：RBAC 贯穿到 LLM 写作层（alice /ask 引用幻觉检查）
├── verify_cache.py      # 阶段4验收：缓存命中 + 精确失效 + 全清 16 项断言
├── CHANGES.md           # 5 阶段改造记录：每阶段改了什么、为什么改、验证结果、已知限制
├── 生产级RAG改造指令集与知识点.md  # 5 阶段指令 + 配套知识点（边做边学）
├── demo_script.md       # 1 分钟 demo 脚本（CLI + HTTP + RBAC + 缓存演示）
├── requirements.txt     # 依赖清单
└── .env.example         # 配置模板（DEEPSEEK_API_KEY/JWT_SECRET/LANGCHAIN_API_KEY/Redis）
```

## 快速开始

### 0. 进入项目根目录

所有命令在 `agent_AutoResearch/` 运行，否则相对路径 `./docs`、`./chroma_data` 会找不到。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

新建 `.env` 文件（已在 `.gitignore`，不要上传 Git）：

```bash
DEEPSEEK_API_KEY=你的密钥             # 必填，调 LLM 用
JWT_SECRET=你的随机字符串              # 建议配，未配会用开发密钥（启动有警告）
LANGCHAIN_API_KEY=ls_xxx              # 可选，配了开 LangSmith 监控，不配自动降级
LANGCHAIN_PROJECT=AutoResearch        # 可选
REDIS_HOST=127.0.0.1                  # 可选，默认 127.0.0.1:6379
REDIS_PORT=6379
```

### 3. 启动 Redis（可选，阶段 4 缓存用）

未装 Redis 也能跑——cache 模块连不上自动降级到只用一级 LRU，不影响业务。

### 4. 启动服务（三选一）

**HTTP 服务模式（主流，生产用这个）**

```bash
uvicorn app:app --reload --port 8000
# 首次启动等十几秒加载 embedding 模型 + 索引 docs/ 知识库
```

看到 `Application startup complete.` 就绪。

**CLI 模式（单问题快速验证）**

```bash
python main.py
```

会问你要研究问题，跑完整流水线，控制台打印报告。

**评估模式**

```bash
python eval/evaluate.py
```

跑 10 道跨领域测试题，输出引用准确率/答案质量。

### 5. 用接口（HTTP 模式启动后）

```bash
# 健康检查（无需登录）
curl http://127.0.0.1:8000/health

# 登录拿 JWT
curl -X POST http://127.0.0.1:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# 跑完整研究流水线（1~3 分钟）
curl -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  --max-time 300 \
  -d '{"question":"猫有哪些生活习性？"}'

# 直接检索知识库（快，专测 RBAC）
curl -X POST http://127.0.0.1:8000/search \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query":"薪资等级","top_k":5}'

# 上传知识库文件（仅 admin）
curl -X POST http://127.0.0.1:8000/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@docs/宠物百科.txt"

# 查缓存规模
curl http://127.0.0.1:8000/cache/stats -H "Authorization: Bearer <token>"

# 手动清缓存（仅 admin）
curl -X DELETE http://127.0.0.1:8000/cache/clear -H "Authorization: Bearer <token>"
```

> **Windows 控制台编码坑**：`curl -d '中文'` 会按 GBK 发送导致 JSON 解析 400。测中文请把 JSON 写成 UTF-8 文件后 `--data-binary @file`，或用 Python/Postman/Invoke-RestMethod。

## 演示用户

写在 [auth.py](auth.py) 里，生产应换数据库 + bcrypt：

| 用户 | 密码 | 角色 | 部门 | 权限 |
|------|------|------|------|------|
| admin | admin123 | admin | — | 全库可见 + 唯一可上传 |
| alice | alice123 | hr | hr | hr 机密 + 公开 |
| bob | bob123 | user | tech | tech 机密 + 公开 |

## 接口清单

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/login` | 无 | 用户名密码换 JWT |
| GET | `/health` | 无 | 健康检查（知识库块数/联网开关/迭代轮数） |
| POST | `/ask` | 登录 | 同步跑完整研究流水线（1~3 分钟），检索范围按 JWT 角色过滤 |
| POST | `/search` | 登录 | 直接检索知识库，专用于验证「不同角色同题检索结果不同」 |
| POST | `/upload` | 仅 admin | 上传知识库文件，立即返 task_id；查询参数 `roles`/`departments` 打权限标签 |
| GET | `/tasks/{id}` | 登录 | 查文件处理进度/状态 |
| GET | `/cache/stats` | 登录 | 看缓存规模（lru_size/redis_enabled/model_version 等） |
| DELETE | `/cache/clear` | 仅 admin | 全量清缓存（LRU + Redis） |

## 验收脚本（持续回归资产）

5 阶段改造产出 3 个验收脚本，累计 50 项断言全过：

| 脚本 | 验证内容 | 通过 |
|------|----------|------|
| `python rbac_e2e_test.py` | RBAC 检索层：401/403 全对、bob 看不到 HR 机密、公开文档全员可见、受限文档连该看的人也看得到（17 项） | 17/17 |
| `python verify_alice_ask.py` | RBAC 贯穿到 LLM 写作层：alice /ask 报告引用幻觉为 0、薪资机密正确引用、tech 机密未泄露 | 通过 |
| `python verify_cache.py` | 缓存命中 + 精确失效 + 全清：retrieve 加速 12x、/ask 加速 148000x、invalidate_doc 按 doc_id 删、DELETE /cache/clear 全清（16 项） | 16/16 |

改 RBAC 或缓存相关代码后建议重跑对应脚本。

## 评估结果

在 10 道跨领域测试题（知识库内 6 题 / 代码执行 2 题 / 知识库外 2 题）上，
以「引用准确率」和「答案质量」两个指标评估：

| 指标 | 优化前（无反思） | 优化后（有反思） |
|------|:---:|:---:|
| 平均引用准确率 | ___% | ___% |
| 平均答案质量（0-100） | ___ | ___ |

> 填上 `python eval/evaluate.py` 跑出的数字即可。基线对比模板见 [eval/results_template.md](eval/results_template.md)。

## 改造记录与学习路径

- **[CHANGES.md](CHANGES.md)**：5 阶段改造全记录——每阶段新建/修改了哪些文件、改了什么、为什么改、验证结果、已知坑。
- **[生产级RAG改造指令集与知识点.md](生产级RAG改造指令集与知识点.md)**：5 阶段指令 + 配套知识点（FastAPI/异步/可观测性/JWT-RBAC/缓存失效），边做边学。

5 阶段建议推进顺序：
1. 阶段 0 FastAPI 工程化（地基，后面全靠它）
2. 阶段 2 LangSmith 监控（最快见效，最先能看到链路图）
3. 阶段 1 异步 + 切块（RAG 质量核心）
4. 阶段 3 RBAC（独立，可插队）
5. 阶段 4 缓存（依赖前面的权限/文件/模型变更点）

> 首次运行会从镜像 `hf-mirror.com` 下载 embedding 模型（已在代码中设置 `HF_ENDPOINT`）。
> 联网搜索默认关闭（国内网络下 DuckDuckGo 被墙）。若网络可用，设环境变量 `ENABLE_WEB_SEARCH=true` 即可开启「本地知识库 + 联网」双路检索。
