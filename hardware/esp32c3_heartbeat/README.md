# RAG 心跳屏（ESP32-C3 + OLED）

把这块小板子做成问答服务的"值班灯"：屏幕上一直显示服务状态、响应延迟和提问速率。
数据通过**蓝牙 BLE** 从电脑推给板子，板子不需要联网、不需要 IP，插 USB 或充电头都能亮着。

```
电脑  scripts/heartbeat_server.py  ──BLE(NUS)──▶  ESP32-C3  ──I2C──▶  SSD1306 OLED
       轮询 /health，5 秒一行                    解析→显示                        128×64
```

> 早期版本带蜂鸣器，实测放弃：模块要用 5V、工作电流 30–40 mA，而 ESP32 的 GPIO
> 最多吸 20 mA 左右。板上的 GND/3V3 又都被 OLED 占了，只能拿 GPIO0 当"假地"，
> 结果蜂鸣器无论信号给高还是给低都长鸣（电流从信号脚自己找回路）。
> 现在的固件**只驱动屏幕**，报警靠屏幕颜色与文字表达。

## 接线

| 板子引脚 | 接到 | 说明 |
|---|---|---|
| GPIO8 | OLED SDA | 默认 I2C（代码里 `I2C_CANDIDATES` 会依次试 8/9、4/5、6/7）|
| GPIO9 | OLED SCL | 同上 |
| 3V3 | OLED VCC | |
| GND | OLED GND | |

如果 OLED 没亮，改 `main.py` 里的 `I2C_CANDIDATES`，或者看串口有没有
`oled ok 0x3c`；没有就是接线或地址（0x3C/0x3D）问题。

## 固件：为什么是 MicroPython

Arduino 的 ESP32 内核需要 673 MB 的 RISC-V 工具链，这台机器的 GitHub
只有 10 KB/s，所有国内镜像都不可用，所以改用 **MicroPython**：固件只有 1.7 MB，
程序全部是 Python，改一行不用重新编译。

已经刷好并跑起来了（原厂固件备份在 `esp32c3-factory-backup.bin`，需要还原时
`esptool write_flash 0x0 esp32c3-factory-backup.bin`）。

要自己重刷：

```powershell
.venv\Scripts\python.exe -m esptool --port COM5 --baud 460800 erase_flash
.venv\Scripts\python.exe -m esptool --port COM5 --baud 460800 write_flash 0x0 ESP32_GENERIC_C3-1.25.0.bin
.venv\Scripts\python.exe -m mpremote connect COM5 fs cp main.py :main.py
.venv\Scripts\python.exe -m mpremote connect COM5 reset
```

## 协议（PC → 板子，一行一条，`\n` 结尾）

```
state=OK;lat=837;qpm=3;chunks=90;mock=0;msg=hello
```

| 字段 | 含义 |
|---|---|
| `state` | `OK` / `WARN` / `DOWN` / `WAIT` |
| `lat` | 最近一次 /health 的响应毫秒数，未知用 `--` |
| `qpm` | 最近 60 秒的真实提问数（取自 `/health` 的 `questions_per_minute`，不含健康检查轮询）|
| `chunks` | 索引里的片段数 |
| `mock` | `1` 表示跑的是 mock 模型（会显示 MOCK! 并警告）|
| `msg` | 一行短消息，屏幕最后一行显示 |

BLE 默认 MTU 只能写 20 字节，所以 PC 端按 20 字节切片顺序写，板子按 `\n` 拼包。

## 状态

| 屏幕 | 触发条件 |
|---|---|
| `OK` | 正常 |
| `WARN` | 延迟 > 4 秒 / 跑的是 mock / 偶发失败一次 |
| `DOWN` | 连续两次拿不到 /health |
| `WAIT` | 还没收到过数据 |
| `LOST` | 超过 15 秒没收到心跳（电脑端断了）|

## 日常使用

1. 服务跑起来（`RAG 一键公网.bat` 或手工 uvicorn）。
2. 双击 `心跳屏.bat` 启动监视器，屏幕每 5 秒刷新一次。
3. 关掉监视器窗口即停；板子会在 15 秒后显示 `LOST`。

排查：

- 板子没广播：串口看有没有 `ble up as RAG-HB`，没有就是 BLE 初始化失败。
- 屏幕不动：串口看有没有 `oled ok 0x3c`，没有就是 I2C 接线问题。
- 屏幕全亮但不显字：帧缓冲要用 `framebuf.MONO_VLSB`（SSD1306 是页结构），
  用 `MONO_HLSB` 会把字形打碎成雪花点，而全 0xFF 仍然能点亮整屏。
- 收不到数据：先 `python -m mpremote connect COM5 fs cat last.txt` 看板子最后收到
  什么；注意 `mpremote` 会打断正在运行的程序，看完记得 `reset`。

## 日志

监视器会把每条采样写成一行 JSON（`logs/heartbeat/heartbeat.jsonl`），但**不是每 5 秒
写一次盘**：先攒在内存里，攒够 50 条或过了 30 秒才落一次盘；遇到 `WARN`/`DOWN` 这类
异常则立刻写，保证故障现场不丢。

写满一个文件就滚动，历史份数固定，所以磁盘占用有硬上限：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--log` | `logs/heartbeat/heartbeat.jsonl` | 日志路径（相对路径按仓库根算） |
| `--log-max-mb` | `2` | 单个文件上限 |
| `--log-keep` | `5` | 保留几份历史（当前 + 5 份 ≈ **12 MB** 封顶） |
| `--log-flush` | `30` | 每多少秒落一次盘 |
| `--log-records` | `50` | 攒够多少条提前落盘 |
| `--no-log` | — | 完全不写文件 |

滚动规则是最老先删：`heartbeat.jsonl` → `heartbeat.1.jsonl` → `…` → 超出 `keep`
的那份直接 `unlink`，所以长期挂着也不会把盘吃满。写盘失败（文件被占用、盘满）不会
让监视器退出，错误计入 `errors`，积压超过 20 批就丢最老的一半，内存同样有上限。

一条日志长这样：

```json
{"ts": "2026-11-20T23:27:16", "state": "WARN", "line": "state=WARN;lat=2640;qpm=1;chunks=0;mock=0;msg=first failure, retrying", "ble": "off", "sample": {"ok": false, "latency_ms": 2640, "error": "ConnectError", ...}}
```

## 文件

- `micropython/main.py` —— 板子程序（BLE 外设 + OLED 渲染 + 状态机）
- `micropython/ssd1306.py` —— 极简 SSD1306 驱动（无第三方依赖）
- `esp32c3_heartbeat.ino` —— 早期 Arduino 版草稿（工具链下载不动，未编译验证，留作参考）
- `../../scripts/heartbeat_server.py` —— PC 端监视器（轮询 /health，走 BLE 推屏）
