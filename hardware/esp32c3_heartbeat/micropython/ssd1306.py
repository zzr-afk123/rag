# Minimal SSD1306 I2C driver for MicroPython (128x64 / 128x32).
# Written for the RAG heartbeat board; no external dependencies.

import framebuf
import time

_SET_CONTRAST = 0x81
_SET_ENTIRE_ON = 0xA4
_SET_NORM_INV = 0xA6
_SET_DISP = 0xAE
_SET_MEM_ADDR = 0x20
_SET_COL_ADDR = 0x21
_SET_PAGE_ADDR = 0x22
_SET_DISP_START_LINE = 0x40
_SET_SEG_REMAP = 0xA0
_SET_MUX_RATIO = 0xA8
_SET_COM_OUT_DIR = 0xC0
_SET_DISP_OFFSET = 0xD3
_SET_COM_PIN_CFG = 0xDA
_SET_DISP_CLK_DIV = 0xD5
_SET_PRECHARGE = 0xD9
_SET_VCOM_DESEL = 0xDB
_SET_CHARGE_PUMP = 0x8D


class SSD1306_I2C(framebuf.FrameBuffer):
    def __init__(self, i2c, addr=0x3C, width=128, height=64):
        self.i2c = i2c
        self.addr = addr
        self.width = width
        self.height = height
        self.pages = height // 8
        self.buffer = bytearray(self.pages * width)
        # SSD1306 的显存是"页"结构：一个字节 = 同一列的 8 个纵向像素。
        # framebuf 必须用 MONO_VLSB 才能对上；用 MONO_HLSB 会让每个字变成雪花点。
        super().__init__(self.buffer, width, height, framebuf.MONO_VLSB)
        self._cmd(
            _SET_DISP | 0x00,
            _SET_MEM_ADDR, 0x00,
            _SET_DISP_START_LINE | 0x00,
            _SET_SEG_REMAP | 0x01,
            _SET_MUX_RATIO, height - 1,
            _SET_COM_OUT_DIR | 0x08,
            _SET_DISP_OFFSET, 0x00,
            _SET_COM_PIN_CFG, 0x02 if height == 32 else 0x12,
            _SET_DISP_CLK_DIV, 0x80,
            _SET_PRECHARGE, 0xF1,
            _SET_VCOM_DESEL, 0x30,
            _SET_CONTRAST, 0xFF,
            _SET_ENTIRE_ON,
            _SET_NORM_INV,
            _SET_CHARGE_PUMP, 0x14,
            _SET_DISP | 0x01,
        )
        self.fill(0)
        self.show()

    def _cmd(self, *cmds):
        for c in cmds:
            self.i2c.writeto(self.addr, bytes((0x00, c & 0xFF)))

    def power(self, on):
        self._cmd(_SET_DISP | (0x01 if on else 0x00))

    def contrast(self, value):
        self._cmd(_SET_CONTRAST, value & 0xFF)

    def invert(self, invert):
        self._cmd(_SET_NORM_INV | (0x01 if invert else 0x00))

    def show(self):
        self._cmd(_SET_COL_ADDR, 0x00, self.width - 1, _SET_PAGE_ADDR, 0x00, self.pages - 1)
        self.i2c.writeto(self.addr, b"\x40" + self.buffer)


def detect(i2c, addrs=(0x3C, 0x3D)):
    """Return the first responding SSD1306 address on this bus, or None."""
    found = set(i2c.scan())
    for a in addrs:
        if a in found:
            return a
    return None
