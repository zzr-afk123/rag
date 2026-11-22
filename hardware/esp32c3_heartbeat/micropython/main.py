# RAG 心跳屏 —— ESP32-C3 + SSD1306 OLED + 蜂鸣器（MicroPython）
#
# 板子是蓝牙从机（BLE 外设，Nordic UART Service）：
#   PC 端 scripts/heartbeat_server.py 每 5 秒写一行状态过来，
#   本机解析后在 OLED 上显示，异常时用蜂鸣器报警。
#
# 一行协议（分号分隔，键值对）：
#   state=OK;lat=837;qpm=1;chunks=90;mock=0;msg=hello
#
# 状态：
#   OK    正常   （绿点、无声音）
#   WARN  警告   （延迟高 / mock 模型 / 偶发失败，短响一声，20 秒内不重复）
#   DOWN  宕机   （连续失败，三连响，每 20 秒重复一次）
#   WAIT  等待   （从没收到过数据）
#   LOST  失联   （超过 15 秒没收到心跳，和三连响一样处理）

import sys
import time

try:
    import bluetooth
    from machine import I2C, Pin, PWM, reset
except ImportError:  # 在 PC 上跑语法检查时会走到这里
    bluetooth = None
    I2C = Pin = PWM = None
    def reset():
        raise SystemExit

try:
    import ssd1306
except ImportError:
    ssd1306 = None

# ---------------------------------------------------------------- 硬件配置
# 本板只做屏幕显示（蜂鸣器已按需求移除）。
I2C_CANDIDATES = ((8, 9), (4, 5), (6, 7))  # (SDA, SCL) 依次尝试
I2C_FREQ = 400_000
SCREEN_W, SCREEN_H = 128, 64

# 屏幕右下角会显示这个版本号：换过固件后先看它有没有变，
# 否则"屏上还是 MOCK!"其实只是板子里跑的是旧代码。
FW_VERSION = "v3"

LOST_MS = 15_000
REMIND_MS = 20_000

# ---------------------------------------------------------------- BLE 定义
NUS_SERVICE = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_RX_CHAR = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"   # PC -> 板子（写）
NUS_TX_CHAR = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   # 板子 -> PC（通知，未用）

_IRQ_CENTRAL_CONNECT = 1
_IRQ_CENTRAL_DISCONNECT = 2
_IRQ_GATTS_WRITE = 3

DEVICE_NAME = "RAG-HB"


def _adv_payload(name):
    payload = bytearray()
    payload += bytes((2, 0x01, 0x06))  # flags: LE general, no BR/EDR
    encoded = name.encode()
    payload += bytes((len(encoded) + 1, 0x09)) + encoded
    return payload


def _uuid(s):
    if bluetooth is None:
        return s
    return bluetooth.UUID(s)


class Screen:
    """包装 OLED，缺屏时静默降级为纯串口输出。"""

    def __init__(self):
        self.dev = None
        self.addr = None
        for sda, scl in I2C_CANDIDATES:
            try:
                i2c = I2C(0, scl=Pin(scl), sda=Pin(sda), freq=I2C_FREQ)
                addr = ssd1306.detect(i2c) if ssd1306 else None
            except Exception as exc:  # noqa: BLE001
                print("i2c", sda, scl, "fail", exc)
                continue
            if addr is None:
                continue
            try:
                self.dev = ssd1306.SSD1306_I2C(i2c, addr, SCREEN_W, SCREEN_H)
            except Exception as exc:  # noqa: BLE001
                print("oled init fail", exc)
                continue
            self.addr = addr
            print("oled ok", hex(addr), "sda", sda, "scl", scl)
            break
        if self.dev is None:
            print("no oled found; running headless")

    def draw(self, view):
        if self.dev is None:
            return
        fb = self.dev
        fb.fill(0)
        # 顶部状态条：反白文字
        fb.fill_rect(0, 0, SCREEN_W, 13, 1)
        fb.text(view["state"], 2, 3, 0)
        if view["ble"]:
            fb.text("BLE", SCREEN_W - 26, 3, 0)
        else:
            fb.text("---", SCREEN_W - 26, 3, 0)
        # 延迟大数字（居中放大 2 倍用两次普通文本拼出来太丑，直接左对齐）
        fb.text(view["latency"], 2, 17, 1)
        fb.text(view["rate"], 70, 17, 1)
        fb.text(view["chunks"], 2, 28, 1)
        fb.text(view["model"], 68, 28, 1)
        # 分隔线
        fb.hline(0, 39, SCREEN_W, 1)
        fb.text(view["beat"], 2, 43, 1)
        # 消息行左边、固件版本右边（版本号用来判断板子里跑的是不是最新代码）
        fb.text(view["msg"], 2, 53, 1)
        fb.text(FW_VERSION, SCREEN_W - 16, 53, 1)
        # 心跳点：每收到一次数据闪一下，靠 view["blink"] 控制
        if view["blink"]:
            fb.fill_rect(SCREEN_W - 8, 44, 6, 6, 1)
        fb.show()


class Heartbeat:
    def __init__(self):
        self.state = "WAIT"
        self.latency = None
        self.qpm = None
        self.chunks = None
        self.mock = False
        self.msg = "waiting for beat"
        self.last_ms = None
        self.blink_until = 0
        self.last_alarm_ms = 0
        self.warned = False

    def feed(self, line):
        data = {}
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                data[k.strip()] = v.strip()
        state = (data.get("state") or "").upper()
        if state in ("OK", "WARN", "DOWN", "WAIT"):
            self.state = state
        if "lat" in data:
            try:
                self.latency = int(data["lat"])
            except ValueError:
                pass
        if "qpm" in data:
            try:
                self.qpm = int(data["qpm"])
            except ValueError:
                pass
        if "chunks" in data:
            try:
                self.chunks = int(data["chunks"])
            except ValueError:
                pass
        if "mock" in data:
            self.mock = data["mock"] in ("1", "true", "True")
        if "msg" in data:
            self.msg = data["msg"][:40]
        self.last_ms = time.ticks_ms()
        self.blink_until = time.ticks_add(self.last_ms, 250)

    def age_ms(self):
        if self.last_ms is None:
            return None
        return time.ticks_diff(time.ticks_ms(), self.last_ms)

    def view(self, ble_connected):
        age = self.age_ms()
        state = self.state
        if age is None:
            state = "WAIT"
        elif age > LOST_MS and state not in ("DOWN",):
            state = "LOST"
        lat = "-- ms" if self.latency is None else "%d ms" % self.latency
        rate = "--/m" if self.qpm is None else "%d/m" % self.qpm
        chunks = "chunks --" if self.chunks is None else "chunks %d" % self.chunks
        # 只回答"模型在不在线"：YES=真实模型可用，NO=mock 或画面已停更。
        # 不再显示 real / MOCK! 这类短词，屏上小字号渲染容易糊。
        if age is None or age > LOST_MS:
            model = "LLM NO"
        else:
            model = "LLM NO" if self.mock else "LLM YES"
        if age is None:
            beat = "beat --"
        else:
            beat = "beat %ds ago" % (age // 1000)
        msg = self.msg if len(self.msg) <= 17 else self.msg[:16] + "~"
        blink = time.ticks_diff(self.blink_until, time.ticks_ms()) > 0
        return {
            "state": state,
            "latency": lat,
            "rate": rate,
            "chunks": chunks,
            "model": model,
            "beat": beat,
            "msg": msg,
            "ble": ble_connected,
            "blink": blink,
        }


class BleLink:
    def __init__(self, on_line):
        self.on_line = on_line
        self.ble = bluetooth.BLE()
        self.ble.active(True)
        self.ble.config(gap_name=DEVICE_NAME)
        self.connections = set()
        self.buffer = b""
        self.rx_handle = None
        service = (
            _uuid(NUS_SERVICE),
            (
                (_uuid(NUS_RX_CHAR), bluetooth.FLAG_WRITE | bluetooth.FLAG_WRITE_NO_RESPONSE),
                (_uuid(NUS_TX_CHAR), bluetooth.FLAG_NOTIFY),
            ),
        )
        ((self.rx_handle, self.tx_handle),) = self.ble.gatts_register_services((service,))
        self.ble.irq(self._irq)
        self._advertise()

    def _advertise(self):
        self.ble.gap_advertise(100_000, adv_data=_adv_payload(DEVICE_NAME))

    @property
    def connected(self):
        return bool(self.connections)

    def _irq(self, event, data):
        if event == _IRQ_CENTRAL_CONNECT:
            conn_handle, _, _ = data
            self.connections.add(conn_handle)
            print("central connected", conn_handle)
        elif event == _IRQ_CENTRAL_DISCONNECT:
            conn_handle, _, _ = data
            self.connections.discard(conn_handle)
            self.buffer = b""
            print("central disconnected", conn_handle)
            self._advertise()
        elif event == _IRQ_GATTS_WRITE:
            conn_handle, attr_handle = data
            try:
                chunk = self.ble.gatts_read(attr_handle)
            except Exception as exc:  # noqa: BLE001
                print("read fail", exc)
                return
            self.buffer += chunk
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                text = line.decode("utf-8", "replace").strip()
                if text:
                    try:
                        self.on_line(text)
                    except Exception as exc:  # noqa: BLE001
                        print("handle fail", exc)


def main():
    screen = Screen()
    beat = Heartbeat()
    link = None

    def on_line(text):
        print("<", text)
        try:
            with open("last.txt", "w") as fh:  # 便于远程排查：留一行最近收到的心跳
                fh.write(text)
        except Exception:  # noqa: BLE001
            pass
        beat.feed(text)

    if bluetooth is not None:
        try:
            link = BleLink(on_line)
            print("ble up as", DEVICE_NAME)
        except Exception as exc:  # noqa: BLE001
            print("ble init fail", exc)

    screen.draw(beat.view(False))
    last_draw = time.ticks_ms()

    while True:
        now = time.ticks_ms()
        view = beat.view(bool(link and link.connected))
        age = beat.age_ms()
        # 屏幕每 200ms 刷一次（保持闪烁点流畅）
        if time.ticks_diff(now, last_draw) > 200:
            screen.draw(view)
            last_draw = now
        # 只做显示：本板不接蜂鸣器，状态靠屏幕颜色与文字表达
        time.sleep_ms(50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("stopped by user")
    except Exception as exc:  # noqa: BLE001
        print("fatal:", exc)
        time.sleep(2)
        reset()
