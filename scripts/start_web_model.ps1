#Requires -Version 5.1
<#
Start the local model (llama-server) and the web-search proxy in front of it.

  .\scripts\start_web_model.ps1                 # model + proxy (default ports 1234 / 1235)
  .\scripts\start_web_model.ps1 -ProxyOnly      # proxy only, model already running

After it prints "Base URL", put that address into any OpenAI-compatible client
(LM Studio: My Models -> the model -> API settings -> base URL).
Close the window (or Ctrl+C) to stop everything this script started.
#>
[CmdletBinding()]
param(
    [int]$LlamaPort = 1234,
    [int]$ProxyPort = 1235,
    [string]$ModelPath = "",
    [switch]$ProxyOnly,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$StepLog = Join-Path $LogDir "web-model-steps.log"

function Write-Step([string]$Text, [string]$Console = "") {
    $stamp = Get-Date -Format "HH:mm:ss"
    Add-Content -Path $StepLog -Value "[$stamp] $Text" -Encoding UTF8
    if ($Console) { Write-Host "[$stamp] $Console" -ForegroundColor Cyan } else { Write-Host "[$stamp] $Text" -ForegroundColor Cyan }
}

function Test-Endpoint([string]$Url) {
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return ($resp.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Find-LlamaServer {
    $base = Join-Path $env:USERPROFILE ".lmstudio\extensions\backends"
    if (-not (Test-Path $base)) { return $null }
    $candidates = Get-ChildItem -Path $base -Directory |
        Where-Object { $_.Name -like "llama.cpp-*" -and $_.Name -notlike "*vulkan*" } |
        Sort-Object @{ Expression = { if ($_.Name -like "*cuda12*") { 0 } else { 1 } }; Ascending = $true },
                    @{ Expression = { $_.Name }; Descending = $true }
    foreach ($dir in $candidates) {
        $exe = Join-Path $dir.FullName "llama-server.exe"
        if (Test-Path $exe) { return @{ Exe = $exe; Dir = $dir.FullName } }
    }
    return $null
}

function Find-CudaRuntime([string]$BackendDir) {
    if ($BackendDir -notlike "*cuda12*") { return $null }
    $vendor = Join-Path (Split-Path -Parent $BackendDir) "vendor\win-llama-cuda12-vendor-v2"
    if (Test-Path $vendor) { return $vendor }
    return $null
}

function Find-Gguf {
    if ($ModelPath -and (Test-Path $ModelPath)) { return $ModelPath }
    $models = Join-Path $env:USERPROFILE ".lmstudio\models"
    if (-not (Test-Path $models)) { return $null }
    $hit = Get-ChildItem -Path $models -Recurse -Filter "*.gguf" |
        Where-Object { $_.Name -like "*qwen3-4b-instruct*" } |
        Sort-Object Length -Descending |
        Select-Object -First 1
    if ($hit) { return $hit.FullName }
    $any = Get-ChildItem -Path $models -Recurse -Filter "*.gguf" | Sort-Object Length -Descending | Select-Object -First 1
    if ($any) { return $any.FullName }
    return $null
}

$modelProc = $null
$proxyProc = $null

try {
    if (-not $ProxyOnly -and -not (Test-Endpoint "http://127.0.0.1:$LlamaPort/v1/models")) {
        $server = Find-LlamaServer
        if (-not $server) { throw "llama-server.exe not found under %USERPROFILE%\.lmstudio\extensions\backends" }
        $gguf = Find-Gguf
        if (-not $gguf) { throw "No .gguf model found under %USERPROFILE%\.lmstudio\models" }

        $cuda = Find-CudaRuntime $server.Dir
        if ($cuda) { $env:PATH = "$cuda;$($server.Dir);$env:PATH" } else { $env:PATH = "$($server.Dir);$env:PATH" }

        Write-Step "Starting local model: $([System.IO.Path]::GetFileName($gguf))" "Starting local model (first load ~10-20s)..."
        $modelProc = Start-Process -FilePath $server.Exe -ArgumentList @(
            "-m", $gguf, "--host", "127.0.0.1", "--port", "$LlamaPort",
            "-c", "8192", "--parallel", "1", "-fa", "on", "-ngl", "99", "--jinja", "--threads", "4"
        ) -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $LogDir "llama-server.out.log") `
            -RedirectStandardError (Join-Path $LogDir "llama-server.err.log")

        $ready = $false
        for ($i = 0; $i -lt 90; $i++) {
            Start-Sleep -Milliseconds 500
            if (Test-Endpoint "http://127.0.0.1:$LlamaPort/v1/models") { $ready = $true; break }
        }
        if (-not $ready) { throw "Model did not become ready in 45s, see logs\llama-server.err.log" }
        Write-Step "Model ready: http://127.0.0.1:$LlamaPort/v1" "Model ready: http://127.0.0.1:$LlamaPort/v1"
    } else {
        Write-Step "Reusing the model already listening on port $LlamaPort" "Reusing model on port $LlamaPort"
    }

    Write-Step "Starting web-search proxy on port $ProxyPort" "Starting web-search proxy..."
    $proxyProc = Start-Process -FilePath (Join-Path $Root ".venv\Scripts\python.exe") -ArgumentList @(
        "-u", "scripts\web_proxy.py", "--port", "$ProxyPort", "--upstream", "http://127.0.0.1:$LlamaPort"
    ) -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir "web-proxy.out.log") `
        -RedirectStandardError (Join-Path $LogDir "web-proxy.err.log")

    $proxyReady = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Endpoint "http://127.0.0.1:$ProxyPort/health") { $proxyReady = $true; break }
    }
    if (-not $proxyReady) { throw "Proxy did not start, see logs\web-proxy.err.log" }

    $base = "http://127.0.0.1:$ProxyPort/v1"
    Set-Clipboard -Value $base
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor DarkGray
    Write-Host " Base URL (copied to clipboard):" -ForegroundColor Gray
    Write-Host "   $base" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor DarkGray
    Write-Host " Put this into LM Studio / any OpenAI-compatible client as base URL." -ForegroundColor Gray
    Write-Host " Header X-Web-Search: off disables searching for one request." -ForegroundColor DarkGray
    Write-Host " Close this window (or Ctrl+C) to stop everything." -ForegroundColor DarkGray
    Write-Host ""

    while ($true) { Start-Sleep -Seconds 2 }
}
finally {
    Write-Step "Stopping processes started by this script" "Stopping..."
    foreach ($proc in @($proxyProc, $modelProc)) {
        if ($null -ne $proc -and -not $proc.HasExited) {
            try { & taskkill /PID $proc.Id /T /F | Out-Null } catch { }
        }
    }
    Write-Host "Bye" -ForegroundColor DarkGray
}
