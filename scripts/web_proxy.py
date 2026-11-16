"""给本地模型加联网能力：一个 OpenAI 兼容的代理，不改模型本身。

用法（模型照旧跑在 1234，代理听 1235）::

    python scripts/web_proxy.py --upstream http://127.0.0.1:1234 --port 1235

然后把客户端的 base_url 从 ``http://127.0.0.1:1234/v1`` 改成
``http://127.0.0.1:1235/v1``，其余（模型名、温度、流式）都不用动。

它做三件事：

1. 判断这句话要不要联网（时效性问题、显式要求"搜一下"、带链接时）；
2. 要联网就检索 + 抓正文，把结果作为【联网资料】注入到系统提示里；
3. 把（被改写过的）请求转发给真正的 llama-server，流式/非流式都原样转发回去。

也可以强制开关：请求头 ``X-Web-Search: always|auto|off``，其中 auto 是默认。

注意：模型文件、权重、模板一概不动；本代理只改「发给模型的提示」这一层。
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import hmac
import ipaddress
import json
import os
import re
import socket
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, quote, urlparse, urljoin

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
DEFAULT_UPSTREAM = os.environ.get("WEB_PROXY_UPSTREAM", "http://127.0.0.1:1234")
DEFAULT_PORT = int(os.environ.get("WEB_PROXY_PORT", "1235"))

#: 一次检索取几条，注入提示时也按这个数截断
MAX_RESULTS = int(os.environ.get("WEB_PROXY_MAX_RESULTS", "4"))
#: 单页正文最多取多少字（再长模型也用不上，还白占上下文）
PAGE_CHARS = int(os.environ.get("WEB_PROXY_PAGE_CHARS", "1200"))
#: 检索结果缓存多久（秒）——同一个问题连问两次不必重复打搜索引擎
CACHE_TTL = float(os.environ.get("WEB_PROXY_CACHE_TTL", "600"))
#: 抓网页的超时（秒）
PAGE_TIMEOUT = float(os.environ.get("WEB_PROXY_PAGE_TIMEOUT", "8"))

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

#: 出现这些词基本可以判定"需要新鲜信息"。宁可多搜一次，也别答成两年前的事。
_TIMELY_WORDS = (
    "最新", "今天", "今日", "现在", "目前", "当前", "最近", "近期", "新闻", "实时",
    "今年", "去年", "明年", "本周", "上周", "本月", "上个月", "天气", "股价", "汇率",
    "油价", "金价", "票房", "比分", "赛程", "排名", "发布", "上线", "版本", "更新",
    "现任", "当选", "任命", "逝世", "去世", "新规", "新政", "政策", "通知", "公告",
)
#: 用户直接要求联网
_WEB_INTENT = (
    "搜一下", "搜搜", "搜索", "查一下", "查查", "上网", "联网", "网上", "百度一下",
    "搜一搜", "帮我查", "最新消息", "现在几点",
)
#: 明显不需要联网的（省一次检索）
_LOCAL_WORDS = ("翻译", "润色", "改写", "写一首", "写一段", "代码", "sql", "正则")
#: 拼关键词式查询时要去掉的口语成分
_QUESTION_FILLER = re.compile(r"(怎么样|怎么|如何|多少|多久|什么|为什么|哪些|哪个|吗|呢|请问|帮我|我想知道|一下)")

_URL_RE = re.compile(r"https?://[^\s）)\]】>\"]+")
_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_RE = re.compile(r"</?(?:strong|b|em|i|u|mark|span|font|a|small|sup|sub)\b[^>]*>", re.I)
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_PARA_RE = re.compile(r"<(p|div|br|li|h[1-6])[^>]*>", re.I)

#: 360 自家的频道页（地图、文库、AI 问答）对"给出处"没意义，直接跳过
_SO_SKIP_HOSTS = ("map.360.cn", "wenku.so.com", "ai.so.com", "www.so.com", "so.com")


def _bigrams(text: str) -> set[str]:
    """中文按二元组切，英文按整词切——够用来判断"这条结果跟问题有没有关系"。"""
    text = _TAG_RE.sub(" ", text or "").lower()
    chars = [ch for ch in text if not ch.isspace()]
    grams = {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}
    grams.update(re.findall(r"[a-z]{3,}", text))
    return grams


def is_relevant(result: SearchResult, query: str, *, threshold: int = 3) -> bool:
    """搜索引擎偶尔会把整句降级成首词去搜（问"郑州大学在哪"回来一堆"郑州市"）。

    这里用查询与标题/摘要的二元组重叠度做一次粗筛：标题必须命中足够多的片段
    （短查询放宽到 1 个），总体重叠达到阈值才留下。宁可少一条，也别把无关资料塞给模型。
    """
    query_grams = _bigrams(query)
    if not query_grams:
        return True
    title_hit = _bigrams(result.title) & query_grams
    need_title = 1 if len(query.strip()) <= 4 else 2
    if len(title_hit) < need_title:
        return False
    hit = _bigrams(f"{result.title} {result.text}") & query_grams
    return len(hit) >= threshold


# --------------------------------------------------------------------------
# 出站访问闸门（SSRF）
# --------------------------------------------------------------------------
#: 代理和 RAG 服务跑在同一台机器上：127.0.0.1:1234 是模型、:8000 是问答服务、
#: 169.254.169.254 是云上临时凭据。而被抓的链接来自搜索引擎结果页，属于外部可控内容。
#: 不加这道闸门，一句"帮我看看 http://127.0.0.1:8000/console"就能把内部页面读进 prompt。
_BLOCKED_CIDRS = (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
    "172.16.0.0/12", "192.0.0.0/24", "192.168.0.0/16", "198.18.0.0/15",
    "::1/128", "fc00::/7", "fe80::/10",
)
_ALLOWED_SCHEMES = ("http", "https")
#: 先按名字拦一遍，不依赖解析结果：``localhost`` 在 hosts 文件被改过、或测试里
#: DNS 被打桩时，"解析到公网"这种判断会失效。
_BLOCKED_HOSTS = ("localhost", "metadata.google.internal", "metadata")
_BLOCKED_HOST_SUFFIXES = (".localhost", ".internal", ".local", ".localdomain")
#: 手动跟随重定向的上限：自动跟随会把闸门绕过（第一跳公网、第二跳回环）
_MAX_REDIRECTS = 3


def _is_blocked_ip(raw: str) -> bool:
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return True  # 认不出来的写法一律不放行
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def is_safe_url(url: str) -> bool:
    """这个地址能不能抓：只允许 http(s)，且主机必须解析到公网地址。

    域名要现场解析（不是只查黑名单字符串），否则 ``localhost``、
    ``internal.example.edu`` 这类"名字看不出问题、解析到内网"的写法会漏过去。
    """
    try:
        parsed = urlparse(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return False
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        # 字面量 IP 不用查 DNS，直接判段（写 ``http://127.0.0.1`` 的人没必要等他解析）
        return not _is_blocked_ip(str(literal))
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return False
    return bool(infos) and not any(_is_blocked_ip(item[4][0]) for item in infos)


def http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float,
    follow: int = 0,
    guard: Any = is_safe_url,
) -> httpx.Response | None:
    """取一页，失败返回 ``None``（联网功能不许把主链路带崩）。

    重定向手动跟随且每一跳都过 ``guard``：自动跟随等于把闸门绕过
    （第一跳是公网正规站点，第二跳跳到 ``http://127.0.0.1:8000``）。

    ``except httpx.HTTPError`` 是不够的：``httpx.InvalidURL`` 直接继承 ``Exception``。
    有些环境的 ``NO_PROXY`` 写成含 ``[::1]`` 的逗号表，httpx 解析它会抛 InvalidURL，
    于是每一次外发都失败——这时退回"不读环境代理"再试一次，比整条联网链路哑掉好。
    """
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"}
    target, query = url, params
    for hop in range(follow + 1):
        if guard is not None and not guard(target):
            return None
        response = None
        for trust_env in (True, False):
            try:
                response = httpx.get(
                    target,
                    params=query,
                    headers=headers,
                    timeout=timeout,
                    follow_redirects=False,
                    trust_env=trust_env,
                )
                break
            except (httpx.HTTPError, httpx.InvalidURL, OSError):
                continue
        if response is None:
            return None
        if hop < follow and response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location", "").strip()
            if not location:
                return None
            target, query = urljoin(target, location), None
            continue
        return response
    return None


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------
@dataclass
class SearchResult:
    """一条检索结果。`text` 先放摘要，抓到正文后会替换成正文节选。"""

    title: str
    url: str
    text: str
    site: str = ""
    fetched: bool = False


@dataclass
class WebContext:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str = ""
    cached: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.results)


def _clean(text: str) -> str:
    """去标签、反转义、压空白。"""
    text = _SCRIPT_RE.sub(" ", text)
    text = _PARA_RE.sub("\n", text)
    # 命中词常被 <strong>/<b> 包起来，若直接当分隔符会切碎词（"郑州 市"），先整段删掉
    text = _INLINE_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    text = text.replace("\u00a0", " ").replace("&ensp;", " ").replace("&nbsp;", " ")
    text = text.replace("&#0183;", "·")
    text = _SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_RE.sub("\n\n", text).strip()


def _unwrap_bing(href: str) -> str:
    """Bing 有时把结果包成 /ck/a?...&u=a1<base64url>，解回真实地址。"""
    if not href:
        return ""
    if href.startswith("/"):
        href = "https://cn.bing.com" + href
    if "bing.com/ck/a" not in href:
        return href
    try:
        token = (parse_qs(urlparse(href).query).get("u") or [""])[0]
        if token.startswith("a1"):
            token = token[2:]
        token += "=" * (-len(token) % 4)
        return base64.urlsafe_b64decode(token).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - 解不出来就放弃这条
        return ""


def parse_bing(html: str, *, limit: int = MAX_RESULTS) -> list[SearchResult]:
    """从 Bing 结果页里挑出「标题 + 链接 + 摘要」。

    只用正则：结果页结构很稳定（`<li class="b_algo">`），为省一个依赖不值得上解析器。
    """
    out: list[SearchResult] = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html, re.S):
        title_match = re.search(r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", block, re.S)
        if not title_match:
            continue
        url = _unwrap_bing(html_lib.unescape(title_match.group(1)))
        title = _clean(title_match.group(2))
        if not url.startswith("http") or not title:
            continue
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.S) or re.search(
            r'<div class="b_caption"[^>]*>(.*?)</div>', block, re.S
        )
        snippet = _clean(snippet_match.group(1)) if snippet_match else ""
        out.append(
            SearchResult(
                title=title[:120],
                url=url,
                text=snippet[:400],
                site=urlparse(url).netloc.replace("www.", ""),
            )
        )
        if len(out) >= limit:
            break
    return out


def bing_search(query: str, *, limit: int = MAX_RESULTS, timeout: float = 12.0) -> list[SearchResult]:
    """走 cn.bing.com 抓结果页。失败返回空列表，让调用方决定重试还是直接说没查到。"""
    resp = http_get(
        "https://cn.bing.com/search",
        params={"q": query, "setlang": "zh-CN", "ensearch": "0"},
        timeout=timeout,
        follow=2,
    )
    if resp is None or resp.status_code != 200:
        return []
    return parse_bing(resp.text, limit=limit)


def parse_so(html: str, *, limit: int = MAX_RESULTS) -> list[SearchResult]:
    """从 360 结果页里挑结果。真实地址在 `data-mdurl`，`href` 只是 JS 跳转。"""
    out: list[SearchResult] = []
    for block in re.findall(r'<li class="res-list"(.*?)</li>', html, re.S):
        title_match = re.search(r"<h3[^>]*>(.*?)</h3>", block, re.S)
        link_match = re.search(r'<a[^>]+href="([^"]+)"', block, re.S)
        if not title_match or not link_match:
            continue
        title = _clean(title_match.group(1))
        mdurl = re.search(r'data-mdurl="([^"]+)"', block)
        href = html_lib.unescape(mdurl.group(1) if mdurl else link_match.group(1))
        if not title or not href.startswith("http"):
            continue
        if any(host in urlparse(href).netloc for host in _SO_SKIP_HOSTS):
            continue
        snippet_match = (
            re.search(r'<span class="res-list-summary"[^>]*>(.*?)</span>', block, re.S)
            or re.search(r'<p class="res-desc"[^>]*>(.*?)</p>', block, re.S)
            or re.search(r'<div class="res-desc"[^>]*>(.*?)</div>', block, re.S)
        )
        snippet = _clean(snippet_match.group(1)) if snippet_match else ""
        out.append(
            SearchResult(
                title=title[:120],
                url=href,
                text=snippet[:400],
                site=urlparse(href).netloc.replace("www.", ""),
            )
        )
        if len(out) >= limit:
            break
    return out


def so_search(query: str, *, limit: int = MAX_RESULTS, timeout: float = 12.0) -> list[SearchResult]:
    """360 搜索兜底。Bing 对"郑州大学在哪里"这类中文问句会降级成只按"郑州"返回。"""
    resp = http_get("https://www.so.com/s", params={"q": query}, timeout=timeout, follow=2)
    if resp is None or resp.status_code != 200:
        return []
    return parse_so(resp.text, limit=limit)


def keyword_query(query: str) -> str:
    """把"郑州今天天气怎么样"压成"郑州 今天 天气"。

    Bing 对整句口语问法经常降级成只按首词检索，换成关键词式反而正常。
    """
    text = _QUESTION_FILLER.sub(" ", query)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())[:60]


def search_web(query: str, *, limit: int = MAX_RESULTS) -> list[SearchResult]:
    """先 Bing，再按相关性筛；不够就用关键词式问法重搜、最后用 360 补。"""
    picked: list[SearchResult] = []
    seen: set[str] = set()

    def take(items: list[SearchResult], against: str) -> None:
        for item in items:
            if not is_relevant(item, against):
                continue
            if item.url in seen:
                continue
            seen.add(item.url)
            picked.append(item)

    variants = [query]
    keywords = keyword_query(query)
    if keywords and keywords != query:
        variants.append(keywords)

    for variant in variants:
        take(bing_search(variant, limit=limit), variant)
        if len(picked) >= 2:
            return picked[:limit]
    if len(picked) < 2:
        take(so_search(keywords or query, limit=limit), keywords or query)
    return picked[:limit]


def fetch_page(url: str, *, timeout: float = PAGE_TIMEOUT, chars: int = PAGE_CHARS) -> str:
    """抓一页正文并截断。只认 text/html，别的（pdf、图片）直接放弃。

    出站前先过 ``is_safe_url``（含每一跳重定向）：这里的 URL 来自搜索结果页，
    是外部可控内容，不做 SSRF 闸门就等于让模型"替我们访问内网"。
    """
    resp = http_get(url, timeout=timeout, follow=_MAX_REDIRECTS)
    if resp is None or resp.status_code != 200:
        return ""
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype.lower():
        return ""
    text = _clean(resp.text[:400_000])
    return text[:chars]


def enrich(results: Sequence[SearchResult], *, workers: int = 4) -> list[SearchResult]:
    """并发抓正文；抓不到就保留摘要，不因为一页失败丢掉整次检索。"""
    if not results:
        return []
    out = list(results)
    queue = list(out)
    lock = threading.Lock()

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                item = queue.pop(0)
            body = fetch_page(item.url)
            if body:
                item.text = body
                item.fetched = True

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(workers, len(out))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=PAGE_TIMEOUT + 3)
    return out


# --------------------------------------------------------------------------
# 判断与拼装
# --------------------------------------------------------------------------
def last_user_text(messages: Iterable[dict[str, Any]]) -> str:
    for message in reversed(list(messages)):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # 多模态：只取文本块
                return " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
    return ""


def needs_web(question: str) -> bool:
    """要不要联网。规则优先：宁可在明显不需要时省一次，也不要在时效问题上答旧闻。"""
    text = question.strip()
    if not text:
        return False
    if _URL_RE.search(text):
        return True
    lowered = text.lower()
    if any(word in lowered for word in _LOCAL_WORDS):
        return False
    if any(word in text for word in _WEB_INTENT):
        return True
    # 年份出现在问句里，多半是在问"今年/今年新出的规定"
    if re.search(r"(19|20)\d{2}\s*年?", text) and any(
        word in text for word in ("新", "改", "发布", "实施", "还有", "吗", "？", "?")
    ):
        return True
    return any(word in text for word in _TIMELY_WORDS)


def build_query(question: str, *, max_chars: int = 80) -> str:
    """把口语问句压成检索串：去掉客套和标点，长句截断。"""
    text = re.sub(r"^(你好|您好|请问|麻烦问一下|想问一下|想咨询一下)[，,：:\s]*", "", question.strip())
    text = re.sub(r"[？?！!。，,；;：:\s]+", " ", text).strip()
    text = text[:max_chars].strip()
    return text or question.strip()[:max_chars]


def render_block(ctx: WebContext) -> str:
    """把检索结果拼成给模型看的资料块。格式稳定，便于模型引用编号。"""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "【联网资料】（代理自动检索，非用户提供；检索时间 " + stamp + "）",
        "说明：这些是刚从互联网抓到的内容，可能过期或有错。用它回答时效性问题时，"
        "在结论后用 [1] [2] 标注来源，并在末尾附一行「来源：标题 — 链接」。"
        "如果这些资料和问题无关，直接说没查到，不要编。",
        "",
    ]
    for index, item in enumerate(ctx.results, start=1):
        lines.append(f"[{index}] {item.title}（{item.site}）")
        lines.append(f"链接：{item.url}")
        body = item.text.strip()
        if body:
            lines.append("内容：" + body.replace("\n", " ")[:PAGE_CHARS])
        lines.append("")
    return "\n".join(lines).strip()


def render_empty(query: str) -> str:
    """搜不到也要说一声，否则模型会拿训练数据里的旧知识硬答。"""
    stamp = time.strftime("%Y-%m-%d %H:%M")
    return (
        f"【联网检索】代理在 {stamp} 替你检索了「{query}」，没有拿到可用结果。\n"
        "请直接告诉用户「没有查到」或「无法获取实时信息」，不要凭记忆编造具体数字、日期或结论。"
    )


def inject(messages: Sequence[dict[str, Any]], block: str) -> list[dict[str, Any]]:
    """把资料块塞进系统提示；没有系统提示就新建一条放在最前面。"""
    out = [dict(message) for message in messages]
    for message in out:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content.rstrip() + "\n\n" + block
            else:
                message["content"] = block
            return out
    out.insert(0, {"role": "system", "content": block})
    return out


class _Cache:
    """极简 LRU：同一个问题十分钟内再问，不再打搜索引擎。"""

    def __init__(self, ttl: float = CACHE_TTL, size: int = 128) -> None:
        self._ttl = ttl
        self._size = size
        self._data: OrderedDict[str, tuple[float, WebContext]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> WebContext | None:
        with self._lock:
            hit = self._data.get(key)
            if not hit:
                return None
            stamp, ctx = hit
            if time.time() - stamp > self._ttl:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return ctx

    def put(self, key: str, ctx: WebContext) -> None:
        with self._lock:
            self._data[key] = (time.time(), ctx)
            self._data.move_to_end(key)
            while len(self._data) > self._size:
                self._data.popitem(last=False)


_CACHE = _Cache()


def research(question: str, *, limit: int = MAX_RESULTS) -> WebContext:
    """检索 + 抓正文，带缓存。"""
    query = build_query(question)
    cached = _CACHE.get(query)
    if cached is not None:
        ctx = WebContext(query=cached.query, results=cached.results, elapsed_ms=0.0, cached=True)
        return ctx

    started = time.perf_counter()
    results = search_web(query, limit=limit)
    if not results:
        # 降级都捞不到东西时再退一步：把整句原样搜一次（口语句有时反而更准）
        results = search_web(question.strip()[:80], limit=limit)
    results = enrich(results)
    ctx = WebContext(query=query, results=results, elapsed_ms=(time.perf_counter() - started) * 1000)
    if results:
        _CACHE.put(query, ctx)
    return ctx


# --------------------------------------------------------------------------
# 决策：这次请求要不要联网
# --------------------------------------------------------------------------
def wants_web(body: dict[str, Any], mode: str, question: str) -> bool:
    """把三条输入合成一个布尔：请求头 > 客户端显式开关 > 问句启发式。"""
    mode = (mode or "auto").lower()
    if mode == "always":
        return True
    if mode == "off":
        return False
    if body.get("web_search") is True:
        return True
    # 客户端自带 tools 时让它自己调工具，代理别替它做主
    if body.get("tools"):
        return False
    return needs_web(question)


def with_web(
    body: dict[str, Any],
    mode: str = "auto",
    *,
    search: Any = None,
) -> tuple[dict[str, Any], WebContext | None]:
    """需要联网时返回「注入过资料的新 body」，否则原样返回。

    单独抽出来是为了能脱网测试：`search` 可替换成假实现。
    """
    searcher = search or research
    messages = body.get("messages") or []
    question = last_user_text(messages)
    if not question.strip() or not wants_web(body, mode, question):
        return body, None
    ctx = searcher(question)
    if not ctx or not ctx.ok:
        # 该联网但没搜到：告诉模型"确实没有"，而不是让它凭记忆答
        data = dict(body)
        data["messages"] = inject(messages, render_empty(build_query(question)))
        return data, ctx
    data = dict(body)
    data["messages"] = inject(messages, render_block(ctx))
    return data, ctx


# --------------------------------------------------------------------------
# HTTP 层
# --------------------------------------------------------------------------
def create_app(upstream: str = DEFAULT_UPSTREAM, *, token: str = "") -> FastAPI:
    """``token`` 非空时，除 ``/health`` 外的每个接口都要带口令。

    代理默认只听 127.0.0.1，看起来不用鉴权；但它能把本机 GPU 的模型算力、
    以及"替访问者抓一个 URL"的出站能力都交出去。一旦有人为了手机演示改成
    ``--host 0.0.0.1/0.0.0.0``，同一 Wi-Fi 里的任何人都能刷它，所以把口令
    做成显式开关（``--token`` / ``WEB_PROXY_TOKEN``）。
    OpenAI 兼容客户端直接把自己的 api_key 填成这个串即可（走 ``Authorization: Bearer``）。
    """
    app = FastAPI(title="LM Studio Web Proxy", version="0.1.0")
    state = {"upstream": upstream.rstrip("/")}
    expected = token.strip()

    @app.middleware("http")
    async def _require_token(request: Request, call_next):  # noqa: ANN001, ANN202
        if not expected or request.url.path == "/health":
            return await call_next(request)
        bearer = request.headers.get("authorization", "")
        provided = bearer[7:].strip() if bearer[:7].lower() == "bearer " else ""
        provided = provided or request.headers.get("x-proxy-token", "").strip()
        if not hmac.compare_digest(provided, expected):
            return JSONResponse(
                {"error": {"message": "联网代理需要口令：用 Authorization: Bearer <口令> 或 X-Proxy-Token 头", "type": "unauthorized"}},
                status_code=401,
            )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "upstream": state["upstream"],
            "max_results": MAX_RESULTS,
            "cache_size": len(_CACHE._data),  # noqa: SLF001 - 只给运维看
        }

    @app.post("/v1/web/search")
    async def web_search(payload: dict[str, Any]) -> dict[str, Any]:
        """调试用：直接看代理搜到了什么。"""
        question = str(payload.get("query") or payload.get("q") or "").strip()
        if not question:
            return {"error": "query 不能为空", "results": []}
        ctx = research(question, limit=int(payload.get("limit") or MAX_RESULTS))
        return {
            "query": ctx.query,
            "elapsed_ms": round(ctx.elapsed_ms, 1),
            "cached": ctx.cached,
            "results": [
                {"title": item.title, "url": item.url, "site": item.site, "text": item.text[:300]}
                for item in ctx.results
            ],
        }

    @app.get("/v1/models")
    async def models() -> Response:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{state['upstream']}/v1/models")
        return Response(resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        mode = (request.headers.get("x-web-search") or "auto").strip().lower()
        body, _ctx = with_web(body, mode)

        if body.get("stream"):
            return StreamingResponse(
                _stream_upstream(state["upstream"], body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{state['upstream']}/v1/chat/completions", json=body)
        return Response(resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

    @app.api_route("/v1/{rest:path}", methods=["GET", "POST"])
    async def passthrough(rest: str, request: Request) -> Response:
        """其它 /v1 接口原样转发，避免客户端因为少一个端点直接不可用。"""
        payload = await request.body()
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.request(
                request.method,
                f"{state['upstream']}/v1/{rest}",
                content=payload,
                headers={"content-type": request.headers.get("content-type", "application/json")},
            )
        return Response(resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

    return app


async def _stream_upstream(upstream: str, body: dict[str, Any]):
    """原样转发上游的 SSE。这里不做解析，避免破坏客户端依赖的格式。"""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{upstream}/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                yield text
                return
            async for chunk in resp.aiter_raw():
                yield chunk


def main(argv: list[str] | None = None) -> int:
    global MAX_RESULTS

    parser = argparse.ArgumentParser(description="给本地模型加联网能力的 OpenAI 兼容代理")
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="llama-server 地址，默认 http://127.0.0.1:1234")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="代理监听端口，默认 1235")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS)
    parser.add_argument(
        "--token",
        default=os.environ.get("WEB_PROXY_TOKEN", ""),
        help="访问口令（也可用 WEB_PROXY_TOKEN）；非空时除 /health 外都要带 Authorization: Bearer <口令>",
    )
    args = parser.parse_args(argv)

    MAX_RESULTS = args.max_results

    import uvicorn

    app = create_app(args.upstream, token=args.token)
    banner = [
        f"联网代理已启动：http://{args.host}:{args.port}/v1",
        f"上游模型：{args.upstream}/v1",
        "把客户端的 base_url 指到上面那个地址即可；请求头 X-Web-Search: off 可以临时关闭联网。",
    ]
    if args.token:
        banner.append("访问口令：已启用（客户端把 api_key 填成这个串就行）")
    elif args.host not in ("127.0.0.1", "localhost", "::1"):
        banner.append(f"警告：监听在 {args.host} 且没有口令，同网段任何人都能刷这个代理——请设 --token")
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手动启动
    sys.exit(main())
