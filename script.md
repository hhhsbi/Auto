# AutoResearch —— 5 分钟项目演示脚本

## 演示提纲

一句话定位：**这是一个多智能体研究系统 + 生产级 RAG 服务**——能给一个研究问题自动拆解、检索、算数据、写报告、自我评审；已经从命令行脚本演进成带 JWT 鉴权、RBAC 权限隔离、两级缓存、LangSmith 监控、Docker 容器化部署的真实可上线 HTTP 服务。

5 分钟讲完四件事：
1. **是什么**（30 秒）—— 五角色 + 反思循环
2. **怎么跑**（90 秒）—— HTTP 服务 + CLI 流水线 + 评估数字
3. **生产级能力**（90 秒）—— RBAC 隔离 + 缓存命中 + 链路监控
4. **数据说话**（30 秒）—— 17/17 + 16/16 验收断言全过

## 准备工作

1. **启动 Redis**（阶段 4 缓存用，可选；没装会自动降级到只用 LRU）。
2. **启动 HTTP 服务**：`uvicorn app:app --port 8000`，等到 `Application startup complete.`（首次等十几秒加载 embedding 模型）。
3. **终端预先 `python main.py`** 跑一个**能触发代码执行**的问题（推荐 Q7 电动车销量），把 `outputs/` 里生成的图表准备好，演示时切过去看。
4. **提前跑三个验收脚本**记下数字：`python rbac_e2e_test.py` → 17/17；`python verify_cache.py` → 16/16；`python verify_alice_ask.py` → alice /ask 引用幻觉 0。
5. **打开 RedisDesktopManager** 连 `127.0.0.1:6379 db0`，跑完缓存演示后能看到 key。
6. **可选**：在 `.env` 配 `LANGCHAIN_API_KEY`，演示 LangSmith 链路树；不配也能跑（自动降级）。

## 演示流程（含操作 + 台词）

### 第一段（0:00-0:30）：是什么

| 时间 | 操作 | 台词 |
|------|------|------|
| 0:00-0:08 | 对着屏幕，先不跑 | "这是 AutoResearch，一个基于 LangGraph 的多智能体深度研究系统。你给它一个研究问题，它自己拆解、检索、算数据、写报告、再自我评审。" |
| 0:08-0:15 | 展示架构图（README.md 里的 LangGraph 流水线图） | "内部五个角色协作：规划器把问题拆成子问题，检索器去向量库里找资料，代码执行器在沙箱里跑 Python 算增长率和画图，写作器出报告，评审器打分并检查引用有没有编造。" |
| 0:15-0:30 | 指着 CHANGES.md 五阶段表 | "它分 5 个阶段演进成了一个真实的生产级 RAG 服务：FastAPI 接口、文件异步入库、LangSmith 监控、RBAC 权限、二级缓存、Docker 部署。今天我重点演示 3 个生产级能力。" |

### 第二段（0:30-2:00）：怎么跑（HTTP + CLI）

| 时间 | 操作 | 台词 |
|------|------|------|
| 0:30-0:45 | 输入 `curl localhost:8000/health` | "先看健康检查——HTTP 服务在跑，知识库有 29 个块。" |
| 0:45-1:00 | `POST /login` admin 拿 token | "登录拿 JWT，三个演示用户：admin 全库可见、alice 是 HR、bob 是 tech。" |
| 1:00-1:15 | 输入问题：`某品牌电动车 2021-2024 年销量分别为 10万、15万、22万、30万辆，请计算同比增长率并绘制柱状图`，POST /ask | "看，我输入一个需要数据分析的问题。注意它不是一个简单问答，而是要先拆解、再算数、再画图。" |
| 1:15-1:30 | 滚动展示终端日志（planner→researcher→code_executor→writer→reviewer） | "五个角色协作：规划器拆问题，检索器找资料，代码执行器在沙箱里跑 Python 算增长率和画图，写作器出报告，评审器打分检查引用。" |
| 1:30-1:40 | 切到 `outputs/` 展示生成的图表 | "它真的跑通了 Python，算出了增长率——22 年 50%、23 年 46.7%、24 年 36.4%——还生成了这张柱状图。" |
| 1:40-1:50 | 展示最终报告里的 `[来源1][来源2]` 引用标记 | "最终报告里每个结论都带来源编号，可追溯，不是瞎编的。" |
| 1:50-2:00 | 收尾 | "CLI 模式 `python main.py` 跑的是同一条流水线，HTTP 模式多了接口鉴权和缓存。" |

### 第三段（2:00-3:30）：生产级能力 1 —— RBAC 权限隔离

| 时间 | 操作 | 台词 |
|------|------|------|
| 2:00-2:10 | 用 alice 和 bob 各调一次 `/search "薪资等级"` | "看 RBAC——同一个问题，不同角色检索到的素材本身就不同。" |
| 2:10-2:25 | 展示 alice 返回 `hr_salary_confidential.txt`，bob 返回空/公开文档 | "alice 是 HR，能看到薪资机密文档；bob 是 tech 部门，看不到。权限过滤发生在向量排序之前——Chroma where 先筛再排，top_k 只在有权看范围内算，不会泄露。" |
| 2:25-2:35 | 跑 `python rbac_e2e_test.py` | "我写了 17 项端到端断言验收这个权限模型——401/403 全对、越权访问全拦、机密文档连该看的人也看得到。" |
| 2:35-2:40 | 展示 17/17 全过 | "17/17 全过。" |
| 2:40-2:50 | 跑 `python verify_alice_ask.py` | "更进一步——RBAC 不只是检索层的事，要贯穿到 LLM 写作层。alice 跑完整 /ask，27 个引用里越界编号是 0，薪资机密正确引用，tech 机密一字未提。" |
| 2:50-3:00 | 展示脚本输出 | "权限隔离从检索层一直到 LLM 输出层——writer 拿不到无权内容，自然无法编造或引用。" |

### 第四段（3:00-4:30）：生产级能力 2 —— 二级缓存与失效

| 时间 | 操作 | 台词 |
|------|------|------|
| 3:00-3:10 | 清缓存 `DELETE /cache/clear`，看 `/cache/stats` lru_size=0 | "看缓存——一级 LRU 在进程内，二级 Redis 跨进程共享。" |
| 3:10-3:25 | alice `/search "猫的生活习性"` 第一次（耗时 X ms），第二次（耗时 X ms） | "同一个查询，第一次走 Chroma 向量检索，第二次直接命中 LRU——加速十几倍。" |
| 3:25-3:35 | 展示 `/cache/stats` lru_retrieve_keys 增长 | "stats 接口能看到缓存规模、命中数、模型版本——key 里就含 role 和 model_version，权限或模型一变，key 自然就不同，等于自动失效。" |
| 3:35-3:50 | 切到 RedisDesktopManager 看 `rag:emb:*:*:hr:5`、`rag:doc:*` 几个 key | "Redis 里能看到 key 命名空间——`rag:emb:{query_sha}:{model_version}:{role}:{top_k}` 是检索结果缓存，`rag:doc:{doc_id}` 是反向索引，文件重传时按这个 SET 精确找到要删的 key。" |
| 3:50-4:10 | admin 重传 `docs/宠物百科.txt` 触发 `invalidate_doc` | "文件更新时按 doc_id 精确失效——不会全清缓存，只删关联到这份文档的 key。" |
| 4:10-4:20 | 看 `/cache/stats` lru_retrieve_keys 从 1 降到 0 | "看，相关 key 被清掉了，下次查询会重新走 Chroma 拿新结果。" |
| 4:20-4:30 | 跑 `python verify_cache.py` 展示 16/16 | "16 项验收全过：retrieve 加速 12 倍、/ask 加速 14 万倍、精确失效链路验证、全清接口验证。" |

### 第五段（4:30-5:00）：生产级能力 3 + 收尾

| 时间 | 操作 | 台词 |
|------|------|------|
| 4:30-4:45 | 配了 LANGCHAIN_API_KEY 的话，切 LangSmith 面板；没配就说"配 key 后可开" | "LangSmith 监控——根 trace 下挂五节点，耗时树 + token 排序直接定位哪个节点最慢、token 最多。未配 key 优雅降级，不影响业务。" |
| 4:45-5:00 | 报数字收尾 | "5 阶段全过——CLI 跑通反思循环后引用准确率从 X% 提到 Y%；RBAC 17/17 + alice 写作层通过；缓存 16/16；Docker 容器化 + CI/CD 流水线就绪。3 个验收脚本作为回归资产留在仓库里，改了 RBAC 或缓存相关代码重跑就行。改造记录在 CHANGES.md。" |

## 常见问答

### 如果被问"反思循环是什么"

"评审器会给报告打分，不合格就打回写作器重写，最多三轮。这就是反思循环——它让系统从'能写'变成'能改'。"

### 如果被问"为什么不直接用 LangChain"

"LangChain 是工具箱，LangGraph 是状态图编排——前者把组件连成链，后者把节点画成图加条件边。反思循环这种'不合格打回重写'的环状流程，状态图更直观，条件边 `should_continue` 一行就搞定。"

### 如果被问"Chroma where 权限过滤真的安全吗"

"权限过滤在向量排序之前——Chroma where 先按 metadata 筛再算距离，top_k 最相关只在有权看范围内计算。不是'检索完再过滤'，所以不会泄露 chunk 内容。但元数据本身是 Base64 不加密，别把密码放进去。"

### 如果被问"缓存怎么避免脏数据"

"四件套：1) 文件更新按 doc_id 反向索引精确删；2) 权限/模型变更 key 里就带 role/model_version，一变 key 不同自然失效；3) `DELETE /cache/clear` 手动清；4) TTL 5 分钟兜底——即使漏删也不会永远脏。"

### 如果被问"为什么 Redis 用 protocol=2"

"项目连的是 Redis 5.x，不支持 RESP3 协议的 HELLO 握手，redis-py 默认 protocol=3 会报错。设 protocol=2 走旧协议绕开。"

### 如果被问"怎么部署上线"

"Dockerfile 多阶段构建 + docker-compose 一键起 app + Redis；GitHub Actions CI/CD 流水线三阶段——pytest 跑通 → 构建镜像推 GHCR → SSH 到服务器 docker compose pull && up，带 30 次健康检查重试。requirements.lock 锁定 153 个包精确版本，构建可复现。"

## 演示注意事项

- **别选"知识库外"的问题**（如天气、诺贝尔奖）做演示，那会返回"无法确认"，演示效果不好。
- **确保 `.env` 密钥有效、`docs/` 已索引**，否则检索结果是空，报告质量会打折扣。
- **Windows 控制台 curl 直接 `-d '中文'` 会按 GBK 发送导致 JSON 解析 400**。测中文请把 JSON 写成 UTF-8 文件后 `--data-binary @file`，或用 Python/Postman。
- **Redis 未装不影响演示**：cache 模块连不上自动降级到只用一级 LRU，第二段「缓存命中」依然能演示，只是少一段 RedisDesktopManager 看 key 的展示。
- **LangSmith 监控未配 key 不影响业务**：tracing.py 两层优雅降级——未装包 `@traceable` 变空装饰器，未配 key 把 TRACING_V2 置 false 避免警告噪音。
- **8000 端口被占用时**：可能是上次 uvicorn 没干净退出留了 TIME_WAIT，换 `--port 8001` 最快，不影响 Redis 6379。
