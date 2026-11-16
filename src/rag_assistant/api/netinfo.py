"""局域网地址探测（服务器模式用）。

不依赖第三方库；两条路子互补：
1. ``gethostbyname_ex`` 拿主机名解析出的所有 IPv4；
2. UDP "连接" 8.8.8.8 只是让内核选一条出口路由（**不会真的发包**），
   拿到的就是主网卡地址——最常用的那条。

过滤掉回环、169.254（DHCP 失败）、以及 docker/wsl 常见的 172.17/172.18 段，
避免把容器网桥地址当成可访问地址打印给用户。
"""

from __future__ import annotations

import socket

_VIRTUAL_PREFIXES = ("172.17.", "172.18.", "172.19.", "172.2", "169.254.")


def _is_usable(ip: str) -> bool:
    if not ip or ip.startswith(("127.", "0.")):
        return False
    if any(ip.startswith(prefix) for prefix in _VIRTUAL_PREFIXES):
        return False
    return True


def lan_ipv4_addresses() -> list[str]:
    """返回本机可被局域网访问的 IPv4 列表（可能为空）。"""
    addresses: set[str] = set()

    for target in ("8.8.8.8", "1.1.1.1"):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((target, 80))
            candidate = probe.getsockname()[0]
            if _is_usable(candidate):
                addresses.add(candidate)
            break
        except OSError:
            continue
        finally:
            probe.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidate = info[4][0]
            if _is_usable(candidate):
                addresses.add(candidate)
    except OSError:
        pass

    return sorted(addresses)


def build_urls(host: str, port: int) -> list[str]:
    """按绑定地址生成可访问 URL。

    ``host`` 为 ``0.0.0.0`` 时返回全部局域网地址；否则只返回该地址。
    """
    if host in {"0.0.0.0", "::", ""}:
        urls = [f"http://{ip}:{port}/" for ip in lan_ipv4_addresses()]
        return urls or [f"http://127.0.0.1:{port}/"]
    return [f"http://{host}:{port}/"]
