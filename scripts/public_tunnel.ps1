#Requires -Version 5.1
<#
  一键公网演示：本地大模型 + 本地 RAG 服务 + Cloudflare 临时隧道
  ------------------------------------------------------------
  · 双击项目根目录的「一键公网.bat」启动
  · 依次拉起：llama-server（本机 Qwen3-4B）→ uvicorn（RAG 服务）→ cloudflared（临时隧道）
  · 窗口里会打印 https://xxx.trycloudflare.com 地址，自动复制到剪贴板并打开浏览器
  · 关闭这个窗口（或 Ctrl+C）= 三个进程一起退出，不留后台进程

  参数：
    -Port 8000             本地端口
    -LlamaPort 1234        本地模型端口
    -ModelPath ""          指定 .gguf；留空自动在 LM Studio 模型目录里找
    -SkipModel             不启动本地模型（此时回落到 settings.yaml 的默认 provider）
    -BindLan               本地服务额外绑定 0.0.0.0，让同 Wi-Fi 的手机也能直接访问
    -StartupTimeoutSeconds 180   等待服务就绪的上限（秒）
    -NoBrowser             不自动打开浏览器
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [int]$LlamaPort = 1234,
    [string]$ModelPath = "",
    [switch]$SkipModel,
    [switch]$BindLan,
    [int]$StartupTimeoutSeconds = 180,
    [switch]$NoHeartbeat,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# 控制台编码交给 bat 里的 chcp（中文系统 936）；PS 5.1 下再改 OutputEncoding 会让中文每个字画两遍。

$LogDir       = Join-Path $ProjectRoot "logs"
$Script:StepLog = Join-Path $LogDir "launcher-steps.log"
$ServerOut    = Join-Path $LogDir "uvicorn-tunnel.out.log"
$ServerErr    = Join-Path $LogDir "uvicorn-tunnel.err.log"
$TunnelOut    = Join-Path $LogDir "cloudflared.out.log"
$TunnelErr    = Join-Path $LogDir "cloudflared.err.log"
$LlamaOut     = Join-Path $LogDir "llama-server.log"
$LlamaErr     = Join-Path $LogDir "llama-server.err.log"
$HeartbeatOut = Join-Path $LogDir "heartbeat.out.log"
$HeartbeatErr = Join-Path $LogDir "heartbeat.err.log"
$UrlFile      = Join-Path $LogDir "tunnel-url.txt"
$PidFile      = Join-Path $LogDir "tunnel.pid.json"
$VenvPython   = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LlamaBackends = Join-Path $env:USERPROFILE ".lmstudio\extensions\backends"
$LlamaModels   = Join-Path $env:USERPROFILE ".lmstudio\models"
$ModelId       = "qwen3-4b-instruct-2507"

# 模型已经下载好了，就别每次启动再去 huggingface 探测版本：
# hf-mirror 经常 429/500，会白等 30–60 秒并往日志里刷一堆无关告警。
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# 控制台只输出 ASCII（中文在某些 PS 5.1 控制台会被重复绘制）；
# 中文原文写进 logs\launcher-steps.log，出问题可以回看。
function Write-Log([string]$Text, [string]$Prefix = "") {
    if (-not $Script:StepLog) { return }
    $stamp = Get-Date -Format "HH:mm:ss"
    Add-Content -Path $Script:StepLog -Value ("[{0}] {1}{2}" -f $stamp, $Prefix, $Text) -Encoding UTF8
}

function Get-ConsoleText([string]$Text, [string]$Ascii) {
    # 这台机器的控制台在子进程同时写屏时会把中文重影（每个字画两遍），
    # 所以启动阶段的消息只打 ASCII；中文原文照样写进 logs\launcher-steps.log。
    # 想看中文提示可以设 RAG_CONSOLE_CN=1（重启窗口生效）。
    if ($Ascii -and $env:RAG_CONSOLE_CN -ne "1") { return $Ascii }
    return $Text
}

function Write-Step([string]$Text, [string]$Console = "") {
    Write-Log $Text
    $shown = Get-ConsoleText $Text $Console
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $shown) -ForegroundColor Cyan
}

function Write-Note([string]$Text, [string]$Console = "") {
    Write-Log $Text "! "
    $shown = Get-ConsoleText $Text $Console
    Write-Host ("  ! {0}" -f $shown) -ForegroundColor Yellow
}

function Find-Cloudflared {
    $candidates = @(
        "C:\Program Files (x86)\cloudflared\cloudflared.exe",
        "C:\Program Files\cloudflared\cloudflared.exe",
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\cloudflared.exe")
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Find-LlamaServer {
    if (-not (Test-Path $LlamaBackends)) { return $null }
    $dirs = Get-ChildItem $LlamaBackends -Directory -Filter "llama.cpp-*" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "(?i)vulkan" -and (Test-Path (Join-Path $_.FullName "llama-server.exe")) }
    $ranked = $dirs | ForEach-Object {
        # CUDA 版优先，其次 AVX2；同类里挑版本号最高的
        $cuda = if ($_.Name -match "(?i)cuda12") { 0 } elseif ($_.Name -match "(?i)cuda") { 1 } else { 2 }
        $m = [regex]::Match($_.Name, "(\d+(?:\.\d+)+)$")
        $ver = if ($m.Success) { [version]$m.Groups[1].Value } else { [version]"0.0" }
        [pscustomobject]@{
            Name    = $_.Name
            Dir     = $_.FullName
            Exe     = (Join-Path $_.FullName "llama-server.exe")
            Cuda    = $cuda
            Version = $ver
        }
    } | Sort-Object Cuda, @{ Expression = "Version"; Descending = $true }
    if (-not $ranked) { return $null }
    return @{ Exe = $ranked[0].Exe; Dir = $ranked[0].Dir; Name = $ranked[0].Name }
}

function Find-CudaRuntime([string]$BackendName, [string]$BackendDir) {
    $vendorRoot = Join-Path (Split-Path -Parent $BackendDir) "vendor"
    if (-not (Test-Path $vendorRoot)) { return $null }
    # cuda12 后端必须配 cuda12 运行时，否则会以 0xC0000135 静默退出
    $pattern = if ($BackendName -match "(?i)cuda12") { "(?i)cuda12" } else { "(?i)cuda" }
    $dirs = Get-ChildItem $vendorRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match $pattern } |
        Where-Object { Get-ChildItem $_.FullName -Filter "cudart64*.dll" -ErrorAction SilentlyContinue }
    if ($dirs) { return ($dirs | Select-Object -First 1).FullName }
    return $null
}

function Find-Gguf([string]$Explicit) {
    if ($Explicit) {
        if (Test-Path $Explicit) { return (Resolve-Path -LiteralPath $Explicit).Path }
        throw "指定的模型文件不存在：$Explicit"
    }
    if (-not (Test-Path $LlamaModels)) { return $null }
    $candidates = Get-ChildItem $LlamaModels -Recurse -Filter "*.gguf" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "(?i)(mmproj|embed)" }
    if (-not $candidates) { return $null }
    $preferred = $candidates | Where-Object { $_.Name -match "(?i)qwen3-4b-instruct-2507" } |
        Sort-Object Length -Descending | Select-Object -First 1
    if ($preferred) { return $preferred.FullName }
    return ($candidates | Sort-Object Length -Descending | Select-Object -First 1).FullName
}

function Test-LocalHealth([int]$TimeoutSec = 3) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec $TimeoutSec
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Test-LlamaHealth([int]$TimeoutSec = 3) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$LlamaPort/v1/models" -UseBasicParsing -TimeoutSec $TimeoutSec
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Stop-Tree($ProcId) {
    if (-not $ProcId) { return }
    if (Get-Process -Id $ProcId -ErrorAction SilentlyContinue) {
        try { & taskkill.exe /PID $ProcId /T /F 2>$null | Out-Null } catch { }
    }
}

# ---------- 前置检查 ----------
if (-not (Test-Path $VenvPython)) {
    throw "找不到虚拟环境：$VenvPython`n请先在项目目录执行一次环境安装。"
}

$cloudflared = Find-Cloudflared
if (-not $cloudflared) {
    throw "找不到 cloudflared（Cloudflare 隧道客户端）。`n安装命令：winget install --id Cloudflare.cloudflared -e"
}

# ---------- 已经在跑就直接告诉他 ----------
if ((Test-Path $PidFile) -and (Test-Path $UrlFile)) {
    try {
        $prev = Get-Content $PidFile -Raw | ConvertFrom-Json
        if (Get-Process -Id $prev.tunnel_pid -ErrorAction SilentlyContinue) {
            $prevUrl = (Get-Content $UrlFile -Raw).Trim()
            Write-Host ""
            Write-Host "  已经有一条演示通道在运行（不是这个窗口起的）：" -ForegroundColor Yellow
            Write-Host "  $prevUrl" -ForegroundColor Green
            Write-Host "  这个地址可以直接用；要重开一条，请先关掉那个窗口（或用任务管理器结束 cloudflared）。" -ForegroundColor Yellow
            Write-Host ""
            if (-not $NoBrowser) { try { Start-Process $prevUrl } catch { } }
            # 别静默退出：双击 .bat 时窗口一闪而过，用户会以为"按钮坏了"
            Write-Host "  按回车关闭这个窗口..." -ForegroundColor DarkGray
            $null = Read-Host
            exit 0
        }
    } catch { }
}
Remove-Item $PidFile, $UrlFile -ErrorAction SilentlyContinue

$server = $null
$tunnel = $null
$model = $null
$reusedServer = $false
$reusedModel = $false

try {
    # ---------- 0. 本地大模型 ----------
    if ($SkipModel) {
        Write-Step "按要求跳过本地模型，使用 settings.yaml 里的默认 provider" "Skipping local model; using the provider from settings.yaml"
    } elseif (Test-LlamaHealth) {
        $reusedModel = $true
        $env:RAG_LLM__PROVIDER = "lmstudio"
        $env:RAG_LLM__MODEL = $ModelId
        $env:RAG_LLM__BASE_URL = "http://127.0.0.1:$LlamaPort/v1"
        # bge-m3 留在 CPU（够快、省显存），重排器走 GPU；启动时后台预热，第一问不再等模型加载
        $env:RAG_EMBEDDING__DEVICE = "cpu"
        $env:RAG_WARMUP = "1"
        Write-Step "本地模型服务已在 127.0.0.1:$LlamaPort 运行，直接复用" "Local model already running on 127.0.0.1:$LlamaPort - reusing"
    } else {
        $llama = Find-LlamaServer
        if (-not $llama) {
            throw "找不到 llama-server.exe。`n请先安装 LM Studio 并下载一个模型，或用 -ModelPath 指定 .gguf。"
        }
        $gguf = Find-Gguf $ModelPath
        if (-not $gguf) {
            throw "找不到本地模型文件（.gguf）。`n请在 LM Studio 里下载模型，或用 -ModelPath 指定路径。"
        }
        $vendor = Find-CudaRuntime $llama.Name $llama.Dir
        if ($vendor) { $env:PATH = "$vendor;$($llama.Dir);$env:PATH" }
        else { $env:PATH = "$($llama.Dir);$env:PATH" }

        Write-Step ("启动本地模型（{0}，首次加载约 10-20 秒）..." -f (Split-Path -Leaf $gguf)) ("Starting local model ({0}, first load ~10-20s)..." -f (Split-Path -Leaf $gguf))
        Remove-Item $LlamaOut, $LlamaErr -ErrorAction SilentlyContinue
        $model = Start-Process -FilePath $llama.Exe -ArgumentList @(
            "-m", "`"$gguf`"",
            "--host", "127.0.0.1", "--port", "$LlamaPort",
            "-c", "4096", "--parallel", "1",
            "-fa", "on", "-ngl", "99", "--jinja", "--threads", "8"
        ) -WorkingDirectory $llama.Dir -NoNewWindow `
          -RedirectStandardOutput $LlamaOut -RedirectStandardError $LlamaErr -PassThru

        $deadline = (Get-Date).AddSeconds(120)
        while ((Get-Date) -lt $deadline) {
            if (Test-LlamaHealth) { break }
            if ($model.HasExited) {
                $null = $model.WaitForExit(3000)
                $code = if ($null -ne $model.ExitCode) { $model.ExitCode } else { "未知" }
                throw "本地模型启动失败（exit $code），详见：$LlamaErr"
            }
            Start-Sleep -Milliseconds 700
        }
        if (-not (Test-LlamaHealth)) {
            throw "本地模型 120 秒内没有就绪，详见：$LlamaErr"
        }
        $env:RAG_LLM__PROVIDER = "lmstudio"
        $env:RAG_LLM__MODEL = $ModelId
        $env:RAG_LLM__BASE_URL = "http://127.0.0.1:$LlamaPort/v1"
        $env:RAG_EMBEDDING__DEVICE = "cpu"
        $env:RAG_WARMUP = "1"
        Write-Step "本地模型就绪：http://127.0.0.1:$LlamaPort/v1" "Model ready: http://127.0.0.1:$LlamaPort/v1"
    }

    # ---------- 1. 本地服务 ----------
    if (Test-LocalHealth) {
        $reusedServer = $true
        Write-Step "本地服务已在 127.0.0.1:$Port 运行，直接复用（关闭本窗口只会关掉隧道）" "RAG service already running on 127.0.0.1:$Port - reusing (closing this window stops only the tunnel)"
    } else {
        Write-Step "启动本地服务..." "Starting RAG service..."
        Write-Note "首次加载 bge-m3 模型需要 1-2 分钟" "First bge-m3 load takes 1-2 min"
        # 只绑 127.0.0.1：外网流量全走 cloudflared，本机不必再对整个局域网开 8000。
        # 需要手机同 Wi-Fi 一起用就加 -BindLan。
        $bind = if ($BindLan) { "0.0.0.0" } else { "127.0.0.1" }
        # 隧道跑在本机，所以 cloudflared 是唯一的"代理"：只信任 127.0.0.1 传来的
        # X-Forwarded-For（真实访客 IP）与 X-Forwarded-Proto（https）。
        # 公网入口默认收紧限流，避免拿到链接的人刷 LLM 额度。
        if (-not $env:RAG_SERVER__RATE_LIMIT_PER_MIN) { $env:RAG_SERVER__RATE_LIMIT_PER_MIN = "30" }
        $server = Start-Process -FilePath $VenvPython `
            -ArgumentList @("-m", "uvicorn", "rag_assistant.api.main:app",
                "--host", $bind, "--port", "$Port", "--workers", "1",
                "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1") `
            -WorkingDirectory $ProjectRoot -NoNewWindow `
            -RedirectStandardOutput $ServerOut -RedirectStandardError $ServerErr -PassThru

        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            if (Test-LocalHealth) { break }
            if ($server.HasExited) {
                throw "本地服务启动失败（exit $($server.ExitCode)），详见：$ServerErr"
            }
            Start-Sleep -Milliseconds 700
        }
        if (-not (Test-LocalHealth)) {
            throw "本地服务在 $StartupTimeoutSeconds 秒内没有就绪，详见：$ServerErr"
        }
    }
    Write-Step "本地服务就绪：http://127.0.0.1:$Port/" "Service ready: http://127.0.0.1:$Port/"

    # ---------- 1.5 心跳屏（ESP32-C3 的小屏，走蓝牙） ----------
    if (-not $NoHeartbeat) {
        Write-Step "启动心跳屏监视（蓝牙推送到 RAG-HB）..." "Starting heartbeat monitor (BLE -> RAG-HB)..."
        Remove-Item $HeartbeatOut, $HeartbeatErr -ErrorAction SilentlyContinue
        $heartbeat = Start-Process -FilePath $VenvPython `
            -ArgumentList @("-u", (Join-Path $ProjectRoot "scripts\heartbeat_server.py"), "--url", "http://127.0.0.1:$Port", "--interval", "5") `
            -WorkingDirectory $ProjectRoot -NoNewWindow `
            -RedirectStandardOutput $HeartbeatOut -RedirectStandardError $HeartbeatErr -PassThru
        Start-Sleep -Seconds 3
        if ($heartbeat.HasExited) {
            Write-Note "心跳屏监视启动失败（exit $($heartbeat.ExitCode)），详见：$HeartbeatErr" "Heartbeat monitor exited early; see $HeartbeatErr"
            $heartbeat = $null
        } else {
            Write-Step "心跳屏已推送（板子没插也不影响问答，日志：logs\heartbeat.out.log）" "Heartbeat running (log: logs\heartbeat.out.log)"
        }
    }

    # ---------- 2. 公网隧道 ----------
    Write-Step "创建 Cloudflare 临时隧道..." "Creating Cloudflare quick tunnel..."
    $urlPattern = 'https://(?<host>[A-Za-z0-9-]+)\.trycloudflare\.com'
    # cloudflared 出错时会打印 api.trycloudflare.com 这类控制面域名，别把它当隧道地址
    $notTunnel = @("api", "www", "developers", "dash")
    $url = $null
    $lastError = ""
    $tunnel = $null
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        if ($attempt -gt 1) {
            Write-Note "隧道没建成，$(10 * $attempt) 秒后重试（第 $attempt/3 次）" "Tunnel attempt $attempt of 3 failed; retrying in $(10 * $attempt)s"
            Start-Sleep -Seconds (10 * $attempt)
        }
        Remove-Item $TunnelOut, $TunnelErr -ErrorAction SilentlyContinue
        $tunnel = Start-Process -FilePath $cloudflared `
            -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate", "--protocol", "http2") `
            -WorkingDirectory $ProjectRoot -NoNewWindow `
            -RedirectStandardOutput $TunnelOut -RedirectStandardError $TunnelErr -PassThru

        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 500
            foreach ($f in @($TunnelErr, $TunnelOut)) {
                if (-not (Test-Path $f)) { continue }
                try { $text = Get-Content $f -Raw -ErrorAction Stop } catch { continue }
                if (-not $text) { continue }
                # 用 [regex]::Match 而不是 -match + $Matches：后者在函数/循环里偶尔拿到 $null
                $m = [regex]::Match($text, $urlPattern)
                if ($m.Success -and ($notTunnel -notcontains $m.Groups["host"].Value.ToLower())) {
                    $url = $m.Value
                    break
                }
            }
            if ($url) { break }
            if ($tunnel.HasExited) {
                try { $lastError = (Get-Content $TunnelErr -Tail 6 -ErrorAction Stop) -join " | " } catch { }
                break
            }
        }
        if ($url) { break }
    }

    if (-not $url) {
        $hint = ""
        if ($lastError -match "429|Tunnelstore") {
            $hint = "`nCloudflare 现在对免费临时隧道限流了（429 Tunnelstore request failed）——同一条网络短时间内建了太多次隧道。" +
                    "`n  ① 过几分钟再点一次；② 换手机热点再试；③ 实在急就先在局域网演示（http://<本机IP>:$Port/）。"
        }
        throw "没能建起公网隧道。$hint`n最后一段 cloudflared 输出：$lastError`n详见：$TunnelErr"
    }
    $url = $url.Trim()

    # ---------- 3. 从公网侧验一次 ----------
    Write-Step "验证公网地址..." "Verifying the public URL..."
    $probe = $null
    try {
        $probe = (& $VenvPython -c @"
import sys, time
try:
    import httpx
except Exception as exc:
    print('ERR no-httpx ' + type(exc).__name__)
    raise SystemExit(0)
url = sys.argv[1] + '/health'
last = 'timeout'
for _ in range(12):
    try:
        print(httpx.get(url, timeout=25.0).status_code)
        break
    except Exception as exc:
        last = type(exc).__name__
        time.sleep(5)
else:
    print('ERR ' + last)
"@ $url 2>&1 | Select-Object -Last 1)
    } catch { }
    if ("$probe".Trim() -eq "200") {
        Write-Step "公网访问正常（/health 200）" "Public URL is healthy (/health 200)"
    } else {
        Write-Note "公网探测返回 '$probe'，隧道可能还在生效，浏览器刷新几次即可" "Public probe returned '$probe' - the tunnel may still be propagating; refresh the browser"
    }

    # ---------- 4. 记录状态、展示地址 ----------
    Set-Content -Path $UrlFile -Value $url -Encoding UTF8
    [ordered]@{
        server_pid  = if ($server) { $server.Id } else { 0 }
        tunnel_pid  = $tunnel.Id
        model_pid   = if ($model) { $model.Id } else { 0 }
        heartbeat_pid = if ($heartbeat) { $heartbeat.Id } else { 0 }
        url         = $url
        started_at  = (Get-Date).ToString("s")
    } | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

    try { Set-Clipboard -Value $url } catch { }

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor DarkGray
    Write-Host "  公网地址（已复制到剪贴板）" -ForegroundColor White
    Write-Host "  $url" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor DarkGray
    Write-Host "  入口     $url（访客按钮在登录页，只开放问答）" -ForegroundColor Gray
    $credFile = Join-Path $LogDir "credentials.txt"
    if (Test-Path $credFile) {
        Write-Host "  部门/管理账号口令见 $credFile（logs 不入库，别贴进群里）" -ForegroundColor Gray
    } else {
        Write-Host "  部门/管理账号：python scripts\manage_users.py list（口令在 init 时打印过一次）" -ForegroundColor Gray
    }
    Write-Host "  收工     删掉 logs\tunnel-url.txt 与这条 URL，临时域名不会一直有效" -ForegroundColor Gray
    Write-Host ""
    if ($SkipModel) {
        Write-Host "  回答模型：settings.yaml 默认（无 Key 时是 mock）" -ForegroundColor Gray
    } else {
        Write-Host "  回答模型：Qwen3-4B-Instruct-2507（本机 llama.cpp，约 49 token/s）" -ForegroundColor Gray
    }
    Write-Host "  第一次检索要加载 reranker，慢一点属正常。" -ForegroundColor Gray
    if ($heartbeat) {
        Write-Host "  心跳屏   已推送状态到 RAG-HB（板子不在也照常问答）" -ForegroundColor Gray
    }
    if ($reusedModel) {
        Write-Host "  注意：本地模型是之前就开着的，关闭本窗口不会关掉它。" -ForegroundColor Yellow
    }
    if ($reusedServer) {
        Write-Host "  注意：本地服务是之前就开着的，关闭本窗口只关隧道。" -ForegroundColor Yellow
    }
    if (-not $reusedServer -or -not $reusedModel) {
        Write-Host "  关闭本窗口（或 Ctrl+C）= 本次启动的进程一起退出。" -ForegroundColor Yellow
    }
    Write-Host ""

    if (-not $NoBrowser) {
        try { Start-Process $url } catch { }
    }

    # ---------- 5. 守着，直到窗口被关 / Ctrl+C / 任一进程退出 ----------
    while ($true) {
        if ($tunnel.HasExited) { Write-Note "隧道进程已退出，演示结束" "The tunnel process exited - demo finished"; break }
        if ($server -and $server.HasExited) { Write-Note "本地服务已退出，演示结束" "The local service exited - demo finished"; break }
        if ($model -and $model.HasExited) { Write-Note "本地模型进程已退出，演示结束" "The local model process exited - demo finished"; break }
        if ($heartbeat -and $heartbeat.HasExited) {
            Write-Note "心跳屏进程退出了（问答不受影响）；想恢复就重新开一个窗口" "Heartbeat monitor exited (service unaffected)"
            $heartbeat = $null
        }
        Start-Sleep -Milliseconds 800
    }
}
finally {
    Write-Host ""
    Write-Step "正在清理..." "Stopping everything..."
    if ($tunnel -and $tunnel.Id) { Stop-Tree $tunnel.Id }
    if ($server -and $server.Id) { Stop-Tree $server.Id }
    if ($model -and $model.Id) { Stop-Tree $model.Id }
    if ($heartbeat -and $heartbeat.Id) { Stop-Tree $heartbeat.Id }
    Remove-Item $PidFile, $UrlFile -ErrorAction SilentlyContinue
    Write-Step "已退出" "Bye"
}
