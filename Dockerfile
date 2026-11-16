# 说明：
# - torch 必须用 CPU 轮子单独装，否则默认 pypi 的 torch 会拉 2.5GB 的 CUDA 依赖；
# - 模型权重**不烧进镜像**（bge-m3 + reranker 约 4.5GB）。首次启动从 hf-mirror 下载到
#   HF_HOME 卷里，之后复用；离线部署请提前把卷灌好（见 README §10）。
# - 必须单 worker：嵌入式 Chroma、进程内会话、进程内同步锁。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/data/hf

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY . .

# 不用 root 跑服务：容器被攻破时少一层提权空间
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/hf /app/logs /app/data/index \
    && chown -R app:app /app /data
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 刻意不加 --forwarded-allow-ips=*：那等于让任何人用 X-Forwarded-For 伪装 IP，
# 限流和登录节流都会被绕过。确实部署在 Nginx/Caddy 后面时，把代理地址写进
# FORWARDED_ALLOW_IPS（或用 server.trust_forwarded_for 由应用层决定是否采信）。
# --workers 1：嵌入式 Chroma + 进程内会话/锁，多 worker 会各持一份索引。
CMD ["uvicorn", "rag_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]
