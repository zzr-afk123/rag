# 快速启动

Windows / PowerShell 下照抄即可。三步：装依赖 → 建索引 → 起服务。

---

## 最快：双击 `start.bat`

不想敲命令就用它。首次运行会自动建 `.venv`、装依赖、建索引、开浏览器，
之后每次双击只是"同步索引 + 启动服务"。

它还会自动判断能力并打印当前模式：

- 装了 `sentence-transformers` 且本机有 bge-m3 权重 → **REAL**（真实向量）
- 否则 → **MOCK**（提示"回答是规则占位"，仅用于跑通链路）
- 找不到 bge-reranker 权重时自动 `RAG_RERANKER__ENABLED=false`，避免首个查询报错

页面里如果索引为空，会出现一个 **「开始构建知识库」** 按钮，点它即可建索引，不用回命令行。

---

## 打开后怎么用（搜索）

服务起来后，浏览器会自动打开 <http://127.0.0.1:8000/>。页面就是搜索框：

1. **第一次**：如果顶部出现「知识库还没有索引」，先点 **开始构建知识库**（等它显示"完成"）；
2. 在底部输入框写问题，**回车发送**（Shift+Enter 换行）；
3. 回答是流式打字的，每条结论后面带 `[1] [2]` 编号；
4. 回答下方点 **引用来源（N 条）** 展开，能看到《文档名》章节、页码和原文摘录；
5. 知识库里没有答案时，会明确回"当前知识库中未找到相关依据…"并给出最相近的资料，
   **不会编造**；
6. 顶部三个徽标：`LLM xxx（mock）`、`文档 N · chunk M`、`手机访问：http://…`；
   右上角两个按钮：**同步知识库**（新文档入库）、**新会话**（清空上下文）。

也可以用接口或命令行搜：

```powershell
# HTTP（同步）
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"session_id\":\"s1\",\"question\":\"在读证明怎么开？\"}"

# 命令行检索调试（打印 vec / bm25 / rrf / rerank 四路分数）
python scripts\search.py "在读证明怎么开" --top-n 3

# 多轮问答（含流式）
python scripts\chat_demo.py --stream
```

> 找不到答案先别怀疑模型，按这个顺序查：`/kb/status` 看 `vector_chunks` 是否 > 0 →
> 文档是否放进了 `data/raw` → 跑一次 `python scripts\sync.py` → 再用 `search.py` 看召回。

---

## 0. 先确认你要哪种模式

| 模式 | 需要什么 | 效果 |
|---|---|---|
| **A. 全 mock**（最快，5 分钟） | 什么都不用额外下 | 链路完全跑通，但回答由规则生成、向量是哈希的，**不代表真实效果** |
| **B. 真实向量 + 关闭重排**（推荐先跑这个） | bge-m3 权重（约 2.2GB） | 检索是真实语义，回答仍需 DeepSeek Key |
| **C. 全真实** | bge-m3 + bge-reranker-v2-m3（约 4.5GB）+ DeepSeek Key | 生产配置 |
| **D. 本地模型** | LM Studio / Ollama 起一个 OpenAI 兼容服务 | 不花钱，但 4B 量化模型很慢且格式不稳，见 README §3.2 |

下面第 4 节给出三种模式各自的启动命令。

---

## 1. 安装依赖

```powershell
cd rag-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# torch 必须用 CPU 轮子，否则默认会拉 2.5GB 的 CUDA 依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-model.txt   # = requirements.txt + sentence-transformers
pip install -e .          # 让 `python -m rag_assistant...` 与 `rag-api` 可用
```

> 不想装 torch / 不打算用真实模型？可以只跑模式 A：
> `pip install -r requirements.txt`（不含 sentence-transformers，因而不会拖进 torch），
> 运行时设 `RAG_FORCE_MOCK=1`。CI 用的就是这一套。

## 2. 配置

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` 里唯一必须填的是：

```
DEEPSEEK_API_KEY=sk-你的key
```

不填也能启动——系统会自动进入 mock LLM 模式（回答是规则拼的，**只用于演示链路**）。
`config/settings.yaml` 是全部参数的默认值，可用 `RAG_<SECTION>__<KEY>` 环境变量覆盖。

## 3. 准备知识库

```powershell
python scripts\gen_sample_docs.py          # 生成示例 docx / xlsx
python scripts\sync.py --rebuild           # 建索引
python scripts\sync.py --status            # 看状态（JSON）
```

`data/raw/` 就是知识库投放目录：**把制度文档丢进去，再跑一次 `sync.py` 即可**。
服务启动后也会按 `sync.interval_seconds`（默认 60s）自动增量同步。

## 4. 启动服务

### 模式 A：全 mock（无需模型、无需 Key）

```powershell
$env:RAG_FORCE_MOCK = "1"
python scripts\sync.py --rebuild
uvicorn rag_assistant.api.main:app --host 127.0.0.1 --port 8000
```

### 模式 B：真实向量 + 关闭重排

```powershell
# 1) 下载 bge-m3（走镜像；脚本已处理镜像的两个坑）
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
python scripts\download_models.py BAAI/bge-m3

# 2) 关掉 reranker（没下它的权重，开着会在首个查询时报错）
$env:RAG_RERANKER__ENABLED = "false"

# 3) 重建索引 + 起服务
python scripts\sync.py --rebuild
uvicorn rag_assistant.api.main:app --host 127.0.0.1 --port 8000
```

### 模式 C：全真实（含重排）

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
python scripts\download_models.py            # bge-m3 + bge-reranker-v2-m3

python scripts\sync.py --rebuild
uvicorn rag_assistant.api.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/> 就是聊天页面，接口文档在 <http://127.0.0.1:8000/docs>。

## 5. Docker 启动（生产向）

```powershell
Copy-Item .env.example .env      # 填 Key
docker compose up -d --build
docker compose logs -f rag
curl http://127.0.0.1:8000/health
```

- 首次启动会把模型下到 `hf-cache` 卷（4.5GB，取决于网络）；
- 只挂载 `./data`、`./logs`、`./reports`，索引与日志都在宿主机上；
- **固定单 worker**（嵌入式 Chroma、会话、同步锁都在进程内）。

## 6. 常用命令

```powershell
python scripts\search.py "在读证明怎么开" --top-n 3     # 检索调试（看 vec/bm25/rrf/rerank 分数）
python scripts\chat_demo.py --stream                   # 多轮问答演示
python scripts\automation_demo.py --interval 5         # 验收"新增文档 N 秒内可检索"
python scripts\evaluate.py --label real                # 评测：Hit@K / MRR / 拒答 / 忠实度
python -m pytest                                       # 测试（全部离线，不需要模型）
```

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 回答是"根据《xxx》…[1]"这种模板句 | 没配 Key，走的是 mock LLM。填 `DEEPSEEK_API_KEY` 后重启 |
| 首个查询报 `ImportError: sentence-transformers` | 用了真实 embedding 但没装依赖：`pip install sentence-transformers`，或设 `RAG_FORCE_MOCK=1` |
| 首个查询报 reranker 加载失败 | 没下 reranker 权重：`$env:RAG_RERANKER__ENABLED="false"` 或下载模型 |
| 下载模型 403 / 401 | 用脚本：它已设置 `HF_ENDPOINT=hf-mirror` 与 `HF_HUB_DISABLE_XET=1`，并跳过会触发 403 的 `imgs/.DS_Store` |
| 端口被占 | 换 `--port 8001`（`config/settings.yaml` 里也可改） |
| 重启后搜不到文档 | memory 向量库不持久化；本项目默认用 Chroma，检查 `config/settings.yaml` 的 `vectorstore.backend` 不是 `memory` |
| 同步时查询返回 409 | 正在提交索引批次，等几秒重试；`sync.read_lock_timeout_s` 可调 |

---

## 8. 手机 / 局域网访问（服务器模式）

默认 `start.bat` 只监听 `127.0.0.1`（只有本机能用）。要让**同一 Wi-Fi 下的手机、平板、
其他电脑**都能打开，用服务器模式：

```powershell
start-server.bat
# 等价于：start.bat --lan
```

它会绑定 `0.0.0.0`，并打印所有可访问地址 + 一个终端二维码：

```
[4/5] Reachable URLs:
      http://10.190.130.253:8000/
      （以上为局域网地址，手机需连同一个 Wi-Fi）

      █▀▀▀▀▀▀▀████▀█▀██▀█▀▀▀▀▀▀▀█
      █ █▀▀▀█ █ █ ▀ ▄▀ ██ █▀▀▀█ █
      ...
```

手机扫这个二维码（或手动输入 URL）就能直接提问。页面已做移动端适配：
单列布局、输入框 16px（避免 iOS 聚焦自动放大）、引用可折叠。

**打不开时按顺序排查**：

1. 手机和电脑是否同一个 Wi-Fi（不能一个连 5G 一个连访客网络）；
2. Windows 防火墙是否放行（管理员 PowerShell 执行一次）：
   `netsh advfirewall firewall add rule name="RAG 8000" dir=in action=allow protocol=TCP localport=8000`
3. 路由器是否开了"AP 隔离/客户端隔离"（开了之后设备之间不能互访）；
4. 换端口：`start.bat --lan --port 8080`；
5. 用 `ipconfig` 确认打印的 IP 就是当前网卡地址（多网卡/VPN 时可能挑错）。

**安全提醒**（必须看）：

- 项目**没有登录鉴权**，局域网内任何人都能提问、也能触发同步；
- **不要**直接把 8000 端口映射到公网（`netsh interface portproxy` / 路由器端口转发）；
- 需要外网访问，用一层带鉴权的反代或隧道，例如 Nginx + Basic Auth、Cloudflare Tunnel、
  Tailscale（组内网），并在反代上开 HTTPS；
- 公网部署还要在网关做限流，否则 DeepSeek 额度会被刷光。

---

## 9. 用本地模型（LM Studio / Ollama）

```powershell
# 1) LM Studio：Developer → Start Server（默认 127.0.0.1:1234）
# 2) 指向它（不需要 Key）
$env:RAG_LLM__PROVIDER = "lmstudio"
$env:RAG_LLM__BASE_URL = "http://127.0.0.1:1234/v1"
$env:RAG_LLM__MODEL    = "你的模型 id"

# 3) 先冒烟，再启动服务
python scripts\llm_smoke.py --model %RAG_LLM__MODEL%
uvicorn rag_assistant.api.main:app --port 8000
```

冒烟脚本会告诉你这个模型能不能用在本系统：**只输出思考没有答案**（Qwen3 系常见）、
改写不输出 JSON、回答不带 `[n]`、无依据还硬答——任一不合格都会显著推高拒答率。
本机实测 4B 思考型模型约 39s/次，不建议作为生产模型。

---

## 9.4 登录与账号（必看）

首次启动前初始化账号（只写一次，之后改密用 passwd）：

```powershell
.venv\Scripts\python.exe scripts\manage_users.py init
.venv\Scripts\python.exe scripts\manage_users.py list
```

`init` 会为每个账号生成一次性随机口令：屏幕上打印一次，同时写进 `logs/credentials.txt`
（`logs/` 已被 `.gitignore` 排除，别把它贴到群里，用完删掉）。账号 id 固定是
`20230001`（学生）、`jwc01` / `xsc01`（部门，可看部门看板）、`admin`（管理员）。
想换成自己定的口令：

```powershell
.venv\Scripts\python.exe scripts\manage_users.py passwd --id 20230001 --password 新密码
```

登录页支持「访客进入」（仅问答）。学生看不到部门看板与运维台，这是服务端强制的。

## 9.5 手机访问 / 公网部署 / 部门看板

| 想做什么 | 怎么做 |
|---|---|
| 手机打开（同一 Wi-Fi） | 双击 `start-server.bat`，按提示放行防火墙，扫终端里的二维码 |
| 公网任何人都能访问 | 见 `deploy/README.md`：域名 + Nginx（`deploy/nginx.conf`）或 Caddy（`deploy/Caddyfile`）+ HTTPS |
| 看部门问题分布 | 浏览器打开 `/dept`（数据接口要 `dept_console.token`） |
| 生成/预览周报 | 看板右上"下载周报" / "立即投递"（默认只落盘 `logs/outbox/*.eml`，不真发） |
| 没有真实流量先看效果 | `python scripts\seed_demo_questions.py --count 80`（清理加 `--clear`） |

公网部署有两个必须做对的点：`/chat/stream` 关反向代理缓冲（否则流式失效）；
uvicorn 带 `--proxy-headers`（否则限流只看到代理 IP）。

## 10. 一键跑评测（真实模型 + 阈值标定）

双击 **`run-eval.bat`**（或 `python -u scripts\real_eval.py`）：

1. 用真实 bge-m3（+ 已下载的 reranker）重建索引；
2. 跑 `data/eval/golden_set.jsonl` 的 10 条题；
3. 输出 `reports/eval-*.md`：Hit@1/3/5、MRR、拒答准确率、要点覆盖、**阈值扫描与推荐值**。

CPU 上大约 5–20 分钟。跑完把报告里的"推荐阈值"写进 `config/settings.yaml` 的
`retrieval.score_threshold`，再重启服务即生效。

---

## 11. 放到公网（手机 / 外地同事）

```powershell
# 1) 先设令牌与限流（强烈建议）
$env:RAG_SERVER__PUBLIC_ACCESS_TOKEN = "自己定一个长串"
$env:RAG_SERVER__RATE_LIMIT_PER_MIN  = "30"

# 2) 启动（监听所有网卡）
uvicorn rag_assistant.api.main:app --host 0.0.0.0 --port 8000

# 3) 另开一个窗口做临时公网隧道（需先装 cloudflared）
cloudflared tunnel --url http://127.0.0.1:8000
#   输出形如 https://xxx.trycloudflare.com —— 把它加上 ?token=你的串 发出去
```

- 公众页：`/`（手机优先）；运维台：`/console`（公网部署请用网关挡住它）；
- **不会被搜索引擎收录**：这个页面没有稳定域名与 sitemap，也不该公开索引；
  要"能被搜到"必须有自己的域名 + HTTPS + 反代 + 鉴权，详见 README §11；
- 局域网内直接用 `start-server.bat` 即可，不用隧道。

---

## 12. 问无关问题却被回答了怎么办？

这类问题现在会被自动拦住，判据是"问句里的实词有没有在资料里出现过"：

- 触发拒答时 `refusal_reason = off_topic`，回答里会写"该问题看起来不属于本知识库的范围"；
- 如果**正常问题也被误拒**（例如换了种说法），把 `config/settings.yaml` 里的
  `retrieval.min_lexical_coverage` 从 0.30 调低（例如 0.2），或设 0 关闭该层；
- 如果**无关问题还能答**，调高到 0.4–0.5；
- 调完重启服务生效；跑 `run-eval.bat` 可以看到"误拒/拒答准确率"的变化。
