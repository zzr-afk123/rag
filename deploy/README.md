# 部署手册：手机访问 → 局域网 → 公网域名（HTTPS）

这份文档只讲"怎么让外面的人打开它"，以及每一步必须注意的那几件事。
应用本身的行为（检索、拒答、引用）不在这里重复，见 README。

---

## 0. 三条链路，先想清楚要哪条

| 场景 | 做法 | 谁能访问 | 要做的事 |
|---|---|---|---|
| 自己电脑上看 | `start.bat` | 只有本机 | 无 |
| **手机 / 同 Wi-Fi 的电脑** | `start-server.bat`（= `start.bat --lan`） | 同一局域网 | 放行防火墙端口 |
| **公网任何人** | 域名 + Nginx/Caddy + HTTPS | 互联网 | 域名、证书、反向代理、访问令牌 |

三条链路用的是同一个服务，**不要为了手机访问去改代码**。

---

## 1. 手机访问（局域网）

### Windows

```powershell
start-server.bat
```

脚本会把 `server.host` 设为 `0.0.0.0`，打印本机局域网地址（含手机扫码用的二维码），
并提示放行防火墙：

```powershell
netsh advfirewall firewall add rule name="RAG 8000" dir=in action=allow protocol=TCP localport=8000
```

手机上打开 `http://<你电脑的局域网IP>:8000/` 即可；部门看板是 `/dept`。
**手机与电脑必须在同一个 Wi-Fi**，校园网如果开了"AP 隔离"，手机也连不上——换热点试。

### 常见问题

- 能打开页面但提问没反应：看是不是服务器在防火墙后面（Windows 会弹"是否允许"）；
- 打开是"知识库还没有内容"：索引没建，先跑一次 `start.bat` 或 `python scripts\sync.py --build`；
- 手机浏览器不支持流式：iOS Safari 14.5+、Chrome 90+ 都支持；旧内核会退化成"一次性出现"。

---

## 2. 公网部署（服务器 + 域名 + HTTPS）

推荐拓扑（单台 VPS 足够，2 vCPU / 4GB 起步，带 GPU 更好）：

```
浏览器 ──HTTPS(443)──> Nginx ──HTTP──> uvicorn 127.0.0.1:8000 ──> Chroma/BM25（本地文件）
```

### 2.1 准备

1. 域名 A 记录指向服务器公网 IP；
2. 安全组/防火墙放行 80、443（**不要**把 8000 暴露到公网）；
3. 服务器装 Python 3.12 + 项目依赖（或直接用 Docker，见 §2.4）。

```bash
git clone <your-repo> /opt/rag-assistant && cd /opt/rag-assistant
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env && vi .env        # 填 DEEPSEEK_API_KEY
```

### 2.2 配置三件"对外"的东西

编辑 `config/settings.yaml`：

```yaml
server:
  host: 127.0.0.1            # 只让本机代理访问；不要写 0.0.0.0
  port: 8000
  public_url: https://zhidu.example.edu.cn   # 控制台会显示这个地址
  public_access_token: ""    # 见下方"要不要令牌"
  rate_limit_per_min: 60

dept_console:
  token: "换成一串随机字符"   # 部门看板数据接口的令牌（/dept/*）
```

**要不要 `public_access_token`？**

- 公网开放给"任何人"：可以留空（页面直接可用），但要接受"别人能刷你的 LLM 额度"；
- 想限制范围：填一个长随机串，然后把 `https://域名/?token=xxx` 发出去，页面会自动保存。

**部门看板一定要设 `dept_console.token`**：里面是全校提问统计，不该公开。

### 2.3 反向代理

二选一：

- Nginx：复制 `deploy/nginx.conf` 到 `/etc/nginx/conf.d/`，改 `server_name` 和证书路径；
  证书用 `certbot --nginx -d zhidu.example.edu.cn`；
- Caddy：把 `deploy/Caddyfile` 并进 `/etc/caddy/Caddyfile`，它会自动申请证书。

**最容易踩的坑**：`/chat/stream` 必须 `proxy_buffering off`，否则"流式"会变成一次性输出。

### 2.4 Docker 部署（可选）

```bash
cp .env.example .env        # 填 Key；也可以留空跑 mock
docker compose up -d --build
docker compose logs -f      # 首次会下载 4.5GB 模型
```

`docker-compose.yml` 里：

- 端口默认 `8000:8000`；如果前面挂 Nginx，改成 `127.0.0.1:8000:8000`，公网只留 443；
- 索引、日志、模型缓存都挂成了卷，重建容器不会丢；
- `--proxy-headers` 已在 Dockerfile 的启动命令里，限流才能拿到真实客户端 IP。

### 2.4.1 账号与权限

上线前先初始化登录账号并改掉示例密码：

```bash
.venv/bin/python scripts/manage_users.py init
.venv/bin/python scripts/manage_users.py list
.venv/bin/python scripts/manage_users.py passwd --id 20230001 --password '足够长的密码'
```

- 学生账号只能问答；部门账号（`--role dept`）才能看 `/dept` 与 `/console`；
- 走 HTTPS 后把 `config/settings.yaml` 的 `auth.cookie_secure` 设为 `true`；
- 登录开启后，共享令牌 `public_access_token` / `dept_console.token` 不再叠加（避免输两次口令）。

### 2.5 后台常驻

非 Docker 部署用 systemd：`deploy/rag-assistant.service`，改一下路径与用户即可。

```bash
sudo systemctl enable --now rag-assistant
journalctl -u rag-assistant -f
```

---

## 3. 上线前检查清单

- [ ] `curl https://域名/health` 返回 `"status":"ok"`，且 `llm_mock=false`（否则 Key 没生效，一直在 mock）；
- [ ] 手机用 4G/5G（不走校园网）打开首页，提问能看到逐字输出；
- [ ] 已初始化账号并改掉示例密码（`scripts/manage_users.py list`）；
- [ ] 学生账号访问 `/dept` 返回 403（用学生号实测一次）；
- [ ] `auth.cookie_secure=true`（HTTPS 环境）；
- [ ] Nginx 配置里 `/chat/stream` 有 `proxy_buffering off`；
- [ ] 8000 端口没有直接暴露到公网（`ss -lntp` 只看到 127.0.0.1:8000）；
- [ ] 备份：`data/`（文档+索引）、`logs/`（问答日志，周报要用）、`config/`；
- [ ] 周报 `report.enabled=false` 或 `dry_run=true`（先预览，别误发给真实部门）。

---

## 4. 更新与回滚

```bash
cd /opt/rag-assistant
git pull
. .venv/bin/activate && pip install -r requirements.txt
python scripts/sync.py --rebuild      # 只有文档/切块策略变了才需要 rebuild
sudo systemctl restart rag-assistant
```

索引与文档在 `data/`，升级代码不会动它们；回滚只需 `git checkout` 上一次的 tag。
换 embedding 模型必须 `--rebuild`（不同模型的向量不能混用，collection 名带指纹隔离）。
