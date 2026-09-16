# ── 阶段 1：构建依赖（装 pip 包到独立目录，精简最终镜像）──
FROM python:3.11-slim AS builder

WORKDIR /build

# 只复制依赖清单，利用 Docker 层缓存（改代码不重装依赖）
# 用 requirements.lock（锁版本，可复现）；保留 requirements.txt 作源清单
COPY requirements.lock requirements.txt ./

# 装到 /install 目录，最终镜像只 COPY 这个目录
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock

# ── 阶段 2：运行时镜像（不含编译工具，更小更安全）──
FROM python:3.11-slim AS runtime

LABEL maintainer="AutoResearch Team"
LABEL description="多智能体深度研究 RAG 服务"

WORKDIR /app

# 从 builder 阶段拷贝已装好的依赖
COPY --from=builder /install /usr/local

# 拷贝项目代码
COPY . .

# 创建数据目录（Chroma 向量库 + SQLite + 上传文件）
RUN mkdir -p /app/docs /app/chroma_data /app/uploads

# 环境变量默认值（生产部署时用 -e 或 docker-compose 覆盖）
ENV DEEPSEEK_API_KEY="" \
    REDIS_HOST="redis" \
    REDIS_PORT="6379" \
    JWT_SECRET="" \
    ENABLE_WEB_SEARCH="false"

EXPOSE 8000

# 健康检查：/health 返回 200 说明服务存活
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# 启动命令（单 worker，配合 docker-compose 的 replicas 横向扩展）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
