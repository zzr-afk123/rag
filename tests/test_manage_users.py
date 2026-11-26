"""``scripts/manage_users.py`` 的回归测试。

存在的理由：GitHub Actions 的 windows-latest runner 控制台是 cp1252，
脚本里 ``print("已写入账号：…")`` 会抛 UnicodeEncodeError，让 CI 的
「Create account store」这一步直接失败（ubuntu 上因为默认 UTF-8 不会暴露）。
这里用 ``PYTHONIOENCODING=cp1252`` 起子进程把那个场景固化下来。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "manage_users.py"


def _run(args: list[str], *, encoding: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = encoding
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def test_init_survives_ascii_console(tmp_path: Path) -> None:
    target = tmp_path / "users.yaml"

    proc = _run(["--file", str(target), "init", "--weak-demo-passwords"], encoding="cp1252")

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert target.exists(), "账号文件没有写出来"
    text = target.read_text(encoding="utf-8")
    assert "张三" in text and "20230001" in text


def test_init_is_idempotent_without_clobbering(tmp_path: Path) -> None:
    target = tmp_path / "users.yaml"

    first = _run(["--file", str(target), "init", "--weak-demo-passwords"], encoding="utf-8")
    assert first.returncode == 0, first.stderr
    before = target.read_text(encoding="utf-8")

    second = _run(["--file", str(target), "init", "--weak-demo-passwords"], encoding="utf-8")

    assert second.returncode == 0, second.stderr
    assert target.read_text(encoding="utf-8") == before
    assert "已存在" in second.stdout
