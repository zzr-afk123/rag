# RAG 心跳屏（ESP32-C3 + OLED + 蜂鸣器）

把这块小板子做成问答服务的"值班灯"：屏幕上一直显示服务状态与响应延迟，
服务挂了就响三声，恢复了就恢复正常状态。数据通过**蓝牙 BLE** 从电脑推给板子，
板子不需要联网、不需要 IP，插 USB 或充电头都能亮着。

```
电脑  scripts/heartbeat_server.py  ──BLE(NUS)──▶  ESP32-C3  ──I2C──▶  SSD1306 OLED
       轮询 /health，5 秒一行                    解析→显示                        128×64
```

## 接线

| 板子引脚 | 接到 | 说明 |
|---|---|---|
| GPIO8 | OLED SDA | 默认 I2C（代码里 `I2C_CANDIDATES` 会依次试 8/9、4/5、6/7）|
| GPIO9 | OLED SCL | 同上 |
| 3V3 / GND | OLED VCC / GND | |
| GPIO0 | 蜂鸣器信号 | 有源蜂鸣器模块一般是低电平响，代码里 `BUZZER_ACTIVE_LOW = True` |
| GND | 蜂鸣器 GND | |

如果 OLED 没亮，改 `main.py` 里的 `I2C_CANDIDATES`，或者看串口有没有
`oled ok 0x3c`；没有就是接线或地址（0x3C/0x3D）问题。

## 固件：为什么是 MicroPython

Arduino 的 ESP32 内核需要 673 MB 的 RISC-V 工具链，这台机器的 GitHub
只有 10 KB/s，所有国内镜像都不可用，所以改用 **MicroPython**：固件只有
1.7 MB，程序全部是 Python，改一行不用重新编译。

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
| `qpm` | 每分钟提问数（60 秒滑窗）|
| `chunks` | 索引里的片段数 |
| `mock` | `1` 表示跑的是 mock 模型（会显示 MOCK! 并警告）|
| `msg` | 一行短消息，屏幕最后一行显示 |

BLE 默认 MTU 只能写 20 字节，所以 PC 端按 20 字节切片顺序写，板子按 `\n` 拼包。

## 状态与报警

| 屏幕 | 触发条件 | 蜂鸣器 |
|---|---|---|
| `OK` | 正常 | 不响 |
| `WARN` | 延迟 > 4 秒 / 跑的是 mock / 偶发失败一次 | 短响一声 |
| `DOWN` | 连续两次拿不到 /health | 三连响，每 20 秒重复 |
| `WAIT` | 还没收到过数据 | 不响 |
| `LOST` | 超过 15 秒没收到心跳（电脑端断了）| 三连响，每 20 秒重复 |

## 日常使用

1. 服务跑起来（`RAG 一键公网.bat` 或手工 uvicorn）。
2. 双击 `心跳屏.bat` 启动监视器，屏幕每 5 秒刷新一次。
3. 关掉监视器窗口即停；板子会在 15 秒后显示 `LOST` 并报警。

排查：

- 板子没广播：串口看有没有 `ble up as RAG-HB`，没有就是 BLE 初始化失败。
- 屏幕不动：串口看有没有 `oled ok 0x3c`，没有就是 I2C 接线问题。
- 收不到数据：先 `python -m mpremote connect COM5 fs cat last.txt` 看板子最后收到
  什么；注意 `mpremote` 会打断正在运行的程序，看完记得 `reset`。
- 蜂鸣器不响：多半是电平极性反了，把 `BUZZER_ACTIVE_LOW` 改成 `False` 再传一次
  `main.py`。

## 文件

- `micropython/main.py` —— 板子程序（BLE 外设 + OLED 渲染 + 蜂鸣器状态机）
- `micropython/ssd1306.py` —— 极简 SSD1306 驱动（无第三方依赖）
- `esp32c3_heartbeat.ino` —— 早期 Arduino 版草稿（工具链下载不动，未编译验证，留作参考）
- `../../scripts/heartbeat_server.py` —— PC 端监视器（轮询 /health，走 BLE 推屏）
