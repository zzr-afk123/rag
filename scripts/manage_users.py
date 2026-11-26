"""账号管理：初始化账号、加人、改密、停用。

用法::

    python scripts\\manage_users.py init                      # 生成账号，随机口令（打印一次）
    python scripts\\manage_users.py init --weak-demo-passwords # 用可记忆的示例口令（仅本机演示）
    python scripts\\manage_users.py list
    python scripts\\manage_users.py add --id 20230123 --name 王五 --role student --password 123456
    python scripts\\manage_users.py passwd --id 20230123 --password 新密码
    python scripts\\manage_users.py disable --id 20230123

说明：
1. YAML 里只存 PBKDF2 哈希，不存明文；密码只在命令行出现一次。
2. 服务在跑时改完即生效（UserStore 会按文件修改时间自动重载），不需要重启。
3. ``init`` 默认给每个账号生成随机口令并写进 ``logs/credentials.txt``（该目录不入库、
   不进镜像），明文只出现那一次；``--weak-demo-passwords`` 是给本机演示留的后门，
   用固定示例口令，启动时 ``UserStore.weak_accounts()`` 会告警。
4. ``config/users.yaml`` 已被 .gitignore / .dockerignore 排除，模板见 ``users.yaml.example``。
"""

from __future__ import annotations

import argparse
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from rag_assistant.auth import ROLE_LABELS, UserStore, hash_password  # noqa: E402
from rag_assistant.config import load_settings  # noqa: E402


def make_console_safe() -> None:
    """让脚本在非 UTF-8 控制台下也能跑完。

    GitHub Actions 的 windows-latest runner 默认是 cp1252，``print("已写入账号：…")``
    会直接抛 ``UnicodeEncodeError`` 让整步失败（ubuntu 上因为默认 UTF-8 不会暴露）。
    这里把两个流设成"遇到编码不了的字符就用 ? 代替"，口令与文件写入都不受影响。
    """

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 - 老版本或已被重定向的流不支持 reconfigure
            pass


SEED = [
    ("20230001", "张三", "student", "", "student123"),
    ("20230002", "李四", "student", "", "student123"),
    ("jwc01", "王老师", "dept", "教务处", "dept123"),
    ("xsc01", "刘老师", "dept", "学生工作部（处）", "dept123"),
    ("admin", "系统管理员", "admin", "", "admin123"),
]

#: 辅音+元音交替拼成可念的音节，再补一段数字：约 40 bit，手机上也敲得动
_CONSONANTS = "bdghklnmnpqrstwxyz"
_VOWELS = "aeiou"


def random_password() -> str:
    """生成一次性的随机口令（``kozi-3741-mifa`` 形态）。"""
    parts = []
    for _ in range(2):
        parts.append(
            secrets.choice(_CONSONANTS)
            + secrets.choice(_VOWELS)
            + secrets.choice(_CONSONANTS)
            + secrets.choice(_VOWELS)
        )
    digits = "".join(secrets.choice("0123456789") for _ in range(4))
    return f"{parts[0]}-{digits}-{parts[1]}"


def _load(path: Path) -> dict:
    if not path.exists():
        return {"users": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not isinstance(data.get("users"), list):
        raise SystemExit(f"{path} 格式不对，应为 users: [...]")
    return data


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def cmd_init(path: Path, *, weak: bool = False) -> int:
    if path.exists():
        print(f"已存在：{path}（未覆盖；要重置请先备份再删除）")
        return 0
    data = {"users": []}
    plaintext: list[tuple[str, str, str, str]] = []
    for user_id, name, role, dept, demo_password in SEED:
        password = demo_password if weak else random_password()
        data["users"].append(
            {
                "id": user_id,
                "name": name,
                "role": role,
                "department": dept,
                "password": hash_password(password),
                "active": True,
            }
        )
        plaintext.append((user_id, name, ROLE_LABELS.get(role, role), password))
    _save(path, data)
    print(f"已写入账号：{path}")
    for user_id, name, role_label, password in plaintext:
        print(f"  {user_id:<10} {name:<6} {role_label:<4} 口令 {password}")
    if weak:
        print("！--weak-demo-passwords 用的是公开示例口令，只可在本机演示用。")
    else:
        _write_credentials(path, plaintext)
    return 0


def _write_credentials(users_path: Path, plaintext: Sequence[tuple[str, str, str, str]]) -> None:
    """把明文口令写进 logs/credentials.txt（该目录被 .gitignore / .dockerignore 排除）。

    随机口令只打印一次，演示现场抄手机容易丢；落一个不入库的文件比"记在脑子里"可靠。
    用完请自行删除或移到更安全的地方。
    """
    out = users_path.parents[1] / "logs" / "credentials.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# 演示账号口令（{stamp}）— 此文件不入库、不进镜像，用完请删除", ""]
    lines += [f"{user_id}\t{name}\t{role_label}\t{password}" for user_id, name, role_label, password in plaintext]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        out.chmod(0o600)
    except OSError:  # pragma: no cover - Windows 上 chmod 作用有限
        pass
    print(f"口令已另存：{out}（logs/ 不入库，演示完请删除）")


def cmd_list(path: Path) -> int:
    store = UserStore(path)
    users = list(store.all())
    if not users:
        print("（没有账号）")
        return 0
    print(f"{'编号':<12}{'姓名':<8}{'角色':<6}{'部门':<16}状态")
    for user in users:
        print(f"{user.id:<12}{user.name:<8}{user.role_label:<6}{user.department or '-':<16}{'启用' if user.active else '停用'}")
    return 0


def cmd_add(path: Path, args: argparse.Namespace) -> int:
    data = _load(path)
    if any(str(item.get("id")) == args.id for item in data["users"]):
        print(f"账号已存在：{args.id}（改密用 passwd）")
        return 1
    data["users"].append(
        {
            "id": args.id,
            "name": args.name or args.id,
            "role": args.role,
            "department": args.dept or "",
            "password": hash_password(args.password),
            "active": True,
        }
    )
    _save(path, data)
    print(f"已添加：{args.id}（{ROLE_LABELS.get(args.role, args.role)}）")
    return 0


def cmd_passwd(path: Path, args: argparse.Namespace) -> int:
    data = _load(path)
    for item in data["users"]:
        if str(item.get("id")) == args.id:
            item["password"] = hash_password(args.password)
            _save(path, data)
            print(f"已修改密码：{args.id}（服务在跑会自动生效）")
            return 0
    print(f"账号不存在：{args.id}")
    return 1


def cmd_toggle(path: Path, args: argparse.Namespace, active: bool) -> int:
    data = _load(path)
    for item in data["users"]:
        if str(item.get("id")) == args.id:
            item["active"] = active
            _save(path, data)
            print(f"已{'启用' if active else '停用'}：{args.id}")
            return 0
    print(f"账号不存在：{args.id}")
    return 1


def main(argv: list[str] | None = None) -> int:
    make_console_safe()
    settings = load_settings(ROOT / "config" / "settings.yaml", root=ROOT)
    default_path = settings.users_file

    parser = argparse.ArgumentParser(description="管理登录账号（config/users.yaml）")
    parser.add_argument("--file", default=str(default_path), help="用户库路径")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="文件不存在时写入账号（默认随机口令）")
    init.add_argument(
        "--weak-demo-passwords",
        action="store_true",
        help="使用公开的示例口令（student123 等），仅限本机演示；启动时会告警",
    )
    sub.add_parser("list", help="列出账号")

    add = sub.add_parser("add", help="新增账号")
    add.add_argument("--id", required=True)
    add.add_argument("--name", default="")
    add.add_argument("--role", default="student", choices=sorted(ROLE_LABELS))
    add.add_argument("--dept", default="", help="部门名称（role=dept 时填写）")
    add.add_argument("--password", required=True)

    pw = sub.add_parser("passwd", help="修改密码")
    pw.add_argument("--id", required=True)
    pw.add_argument("--password", required=True)

    for name in ("disable", "enable"):
        cmd = sub.add_parser(name, help=f"{'停用' if name == 'disable' else '启用'}账号")
        cmd.add_argument("--id", required=True)

    args = parser.parse_args(argv)
    path = Path(args.file)

    if args.command == "init":
        return cmd_init(path, weak=bool(getattr(args, "weak_demo_passwords", False)))
    if args.command == "list":
        return cmd_list(path)
    if args.command == "add":
        return cmd_add(path, args)
    if args.command == "passwd":
        return cmd_passwd(path, args)
    if args.command == "disable":
        return cmd_toggle(path, args, False)
    if args.command == "enable":
        return cmd_toggle(path, args, True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
