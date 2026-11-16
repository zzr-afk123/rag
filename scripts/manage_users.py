"""账号管理：初始化示例账号、加人、改密、停用。

用法::

    python scripts\\manage_users.py init                      # 仅当文件不存在时写入示例账号
    python scripts\\manage_users.py list
    python scripts\\manage_users.py add --id 20230123 --name 王五 --role student --password 123456
    python scripts\\manage_users.py add --id jwc02 --name 李老师 --role dept --dept 教务处 --password 123456
    python scripts\\manage_users.py passwd --id 20230123 --password 新密码
    python scripts\\manage_users.py disable --id 20230123

说明：
1. YAML 里只存 PBKDF2 哈希，不存明文；密码只在命令行出现一次。
2. 服务在跑时改完即生效（UserStore 会按文件修改时间自动重载），不需要重启。
3. 初始账号的密码是示例值，上线前务必用 passwd 改掉。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from rag_assistant.auth import ROLE_LABELS, UserStore, hash_password  # noqa: E402
from rag_assistant.config import load_settings  # noqa: E402

SEED = [
    ("20230001", "张三", "student", "", "student123"),
    ("20230002", "李四", "student", "", "student123"),
    ("jwc01", "王老师", "dept", "教务处", "dept123"),
    ("xsc01", "刘老师", "dept", "学生工作部（处）", "dept123"),
    ("admin", "系统管理员", "admin", "", "admin123"),
]


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


def cmd_init(path: Path) -> int:
    if path.exists():
        print(f"已存在：{path}（未覆盖；要重置请先备份再删除）")
        return 0
    data = {"users": []}
    for user_id, name, role, dept, password in SEED:
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
    _save(path, data)
    print(f"已写入示例账号：{path}")
    for user_id, name, role, _dept, password in SEED:
        print(f"  {user_id:<10} {name:<6} {ROLE_LABELS.get(role, role):<4} 初始密码 {password}")
    print("上线前请用 passwd 子命令改掉这些示例密码。")
    return 0


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
    settings = load_settings(ROOT / "config" / "settings.yaml", root=ROOT)
    default_path = settings.users_file

    parser = argparse.ArgumentParser(description="管理登录账号（config/users.yaml）")
    parser.add_argument("--file", default=str(default_path), help="用户库路径")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="文件不存在时写入示例账号")
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
        return cmd_init(path)
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
