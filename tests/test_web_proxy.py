"""联网代理：判断、解析、提示注入与缓存。

这些用例全部离线：检索和抓网页都被替换成假实现，跑测试不需要外网。
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import web_proxy
from web_proxy import (
    SearchResult,
    WebContext,
    build_query,
    inject,
    needs_web,
    parse_bing,
    render_block,
    research,
    wants_web,
    with_web,
)

SAMPLE_BING = """
<ol id="b_results">
<li class="b_algo">
  <h2><a href="https://www.zzu.edu.cn/" target="_blank">郑州大学官网</a></h2>
  <div class="b_caption"><p class="b_lineclamp2">郑州大学是国家「211 工程」重点建设高校&ensp;&#0183;&ensp;现有全日制本科生 5 万余人。</p></div>
</li>
<li class="b_algo">
  <h2><a href="/ck/a?u=a1aHR0cHM6Ly9uZXdzLmV4YW1wbGUuY29tL3p6dQ" target="_blank">某新闻：郑州大学新增专业</a></h2>
  <div class="b_caption"><p>2026 年新增三个本科专业。</p></div>
</li>
<li class="b_algo"><h2>没有链接的块</h2><p>应当被跳过</p></li>
</ol>
"""


@pytest.fixture()
def fake_results() -> list[SearchResult]:
    return [
        SearchResult(
            title="郑州大学官网",
            url="https://www.zzu.edu.cn/",
            text="郑州大学是国家「211 工程」重点建设高校，现有全日制本科生 5 万余人。",
            site="zzu.edu.cn",
            fetched=True,
        ),
        SearchResult(
            title="新增专业新闻",
            url="https://news.example.com/zzu",
            text="2026 年新增三个本科专业。",
            site="news.example.com",
        ),
    ]


# --------------------------------------------------------------------------
# 判断
# --------------------------------------------------------------------------
SAMPLE_SO = """
<ul class="result">
<li class="res-list" data-result-pos="3"><h3 class="res-title"><a href="https://www.so.com/link?m=abc" data-mdurl="https://www.gaokao.cn/school/62/qa/94364">高考问答:<em>郑州大学地址</em>在哪里?</a></h3>
<div class='res-rich'><span class="res-list-summary">郑州大学法定住所地为河南省郑州市科学大道100号。</span></div></li>
<li class="res-list"><h3 class="res-title"><a href="https://map.360.cn/?k=zzu">在郑州市搜索郑州大学 - 360地图</a></h3><p class="res-desc">地图结果</p></li>
<li class="res-list"><h3 class="res-title"><a href="https://www.example.com/x">郑州大学招生简章</a></h3><p class="res-desc">2026 年招生简章已发布。</p></li>
</ul>
"""


def test_keyword_query_strips_fillers() -> None:
    from web_proxy import keyword_query

    assert keyword_query("郑州今天天气怎么样？") == "郑州今天天气"
    assert keyword_query("帮我看一下休学最长能休多久呢") == "看 休学最长能休"


def test_search_web_retries_with_keyword_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    good = [
        SearchResult(title="郑州天气预报", url="https://weather.com.cn/zz", text="郑州今天阴，8℃"),
        SearchResult(title="郑州今天天气 24 小时", url="https://tianqi.2345.com/zz", text="郑州今天天气逐小时播报"),
    ]

    def fake_bing(query: str, *, limit: int = 4) -> list[SearchResult]:
        # 整句问法被搜索引擎降级，只有关键词式才返回正常结果
        return good if query == "郑州今天天气" else []

    monkeypatch.setattr(web_proxy, "bing_search", fake_bing)
    monkeypatch.setattr(web_proxy, "so_search", lambda query, limit=4: [])
    picked = web_proxy.search_web("郑州今天天气怎么样", limit=4)
    assert len(picked) == 2
    assert picked[0].url == "https://weather.com.cn/zz"


def test_clean_keeps_words_together() -> None:
    # 命中词被 <strong> 包着时不能切出空格，否则"郑州市"会变成"郑州 市"
    block = '<li class="b_algo"><h2><a href="https://a.cn/x"><strong>郑州</strong>市</a></h2><p>郑州<em>大学</em></p></li>'
    parsed = parse_bing(block, limit=1)[0]
    assert parsed.title == "郑州市"
    assert parsed.text == "郑州大学"


def test_is_relevant_drops_offtopic_results() -> None:
    tourism = SearchResult(title="郑州景点排行榜前十名", url="https://x/1", text="河南郑州旅游攻略")
    school = SearchResult(title="郑州大学地址在哪里", url="https://x/2", text="科学大道 100 号")
    query = "郑州大学在哪里"
    assert web_proxy.is_relevant(school, query) is True
    assert web_proxy.is_relevant(tourism, query) is False


def test_so_search_uses_mdurl_and_skips_internal_hosts() -> None:
    results = web_proxy.parse_so(SAMPLE_SO, limit=5)
    assert [item.title for item in results] == ["高考问答:郑州大学地址在哪里?", "郑州大学招生简章"]
    assert results[0].url == "https://www.gaokao.cn/school/62/qa/94364"
    assert "科学大道" in results[0].text
    # 360 自家的地图页对"给出处"没意义
    assert all("map.360.cn" not in item.url for item in results)


def test_search_web_falls_back_to_360(monkeypatch: pytest.MonkeyPatch) -> None:
    junk = [SearchResult(title="郑州市_百度百科", url="https://baike/zz", text="郑州市是河南省会")]
    good = [SearchResult(title="郑州大学地址", url="https://zzu.edu.cn/a", text="科学大道 100 号")]
    monkeypatch.setattr(web_proxy, "bing_search", lambda query, limit=4: junk)
    monkeypatch.setattr(web_proxy, "so_search", lambda query, limit=4: good)
    picked = web_proxy.search_web("郑州大学在哪里", limit=4)
    assert [item.url for item in picked] == ["https://zzu.edu.cn/a"]


def test_search_web_keeps_bing_when_relevant(monkeypatch: pytest.MonkeyPatch) -> None:
    good = [
        SearchResult(title="郑州大学官网", url="https://zzu.edu.cn/", text="郑州大学"),
        SearchResult(title="郑州大学招生网", url="https://zs.zzu.edu.cn/", text="招生"),
    ]
    monkeypatch.setattr(web_proxy, "bing_search", lambda query, limit=4: good)
    monkeypatch.setattr(web_proxy, "so_search", lambda query, limit=4: pytest.fail("不该走兜底"))
    assert len(web_proxy.search_web("郑州大学在哪里", limit=4)) == 2


@pytest.mark.parametrize(
    "question",
    [
        "郑州今天天气怎么样？",
        "2026 年最新的休学政策有什么变化？",
        "帮我搜一下郑州大学新校区地址",
        "看看这个页面讲了什么 https://www.zzu.edu.cn/",
        "现任校长是谁",
    ],
)
def test_needs_web_yes(question: str) -> None:
    assert needs_web(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "帮我翻译这句话：今天天气不错",
        "写一段 Python 快排",
        "把这段话润色一下",
        "",
    ],
)
def test_needs_web_no(question: str) -> None:
    assert needs_web(question) is False


def test_wants_web_respects_switch_and_tools() -> None:
    body = {"messages": [{"role": "user", "content": "郑州今天天气怎么样？"}]}
    assert wants_web(body, "always", "郑州今天天气怎么样？") is True
    assert wants_web(body, "off", "郑州今天天气怎么样？") is False
    assert wants_web({**body, "web_search": True}, "auto", "你好") is True
    # 客户端自己带了工具，代理不要抢活
    assert wants_web({**body, "tools": [{"type": "function"}]}, "auto", "郑州今天天气怎么样？") is False


def test_build_query_strips_smalltalk_and_punctuation() -> None:
    assert build_query("请问，郑州大学 2026 年放假安排是怎样的？") == "郑州大学 2026 年放假安排是怎样的"
    assert len(build_query("字" * 200)) == 80


# --------------------------------------------------------------------------
# 解析与拼装
# --------------------------------------------------------------------------
def test_parse_bing_extracts_title_url_and_snippet() -> None:
    results = parse_bing(SAMPLE_BING, limit=5)
    assert [item.title for item in results] == ["郑州大学官网", "某新闻：郑州大学新增专业"]
    assert results[0].url == "https://www.zzu.edu.cn/"
    assert "211" in results[0].text
    # 没有链接的块被跳过，而不是变成空结果
    assert all(item.url.startswith("http") for item in results)
    # Bing 的 /ck/a 跳转被还原成真实地址
    assert results[1].url == "https://news.example.com/zzu"


def test_render_block_numbers_sources_and_urls(fake_results: list[SearchResult]) -> None:
    block = render_block(WebContext(query="郑州大学", results=fake_results))
    assert "[1] 郑州大学官网（zzu.edu.cn）" in block
    assert "链接：https://www.zzu.edu.cn/" in block
    assert "[2] 新增专业新闻" in block
    assert "检索时间" in block


def test_inject_appends_to_system_prompt(fake_results: list[SearchResult]) -> None:
    block = render_block(WebContext(query="q", results=fake_results))
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "郑州大学怎么样"},
    ]
    out = inject(messages, block)
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("你是助手。")
    assert "【联网资料】" in out[0]["content"]
    # 原消息不被就地修改
    assert messages[0]["content"] == "你是助手。"


def test_inject_creates_system_message_when_missing(fake_results: list[SearchResult]) -> None:
    block = render_block(WebContext(query="q", results=fake_results))
    out = inject([{"role": "user", "content": "问"}], block)
    assert out[0]["role"] == "system"
    assert out[1] == {"role": "user", "content": "问"}


def test_with_web_keeps_body_when_no_search_needed() -> None:
    body = {"messages": [{"role": "user", "content": "写一段 Python 快排"}]}
    data, ctx = with_web(body, "auto", search=lambda q: pytest.fail("不该检索"))
    assert ctx is None
    assert data is body


def test_with_web_injects_when_search_hits(fake_results: list[SearchResult]) -> None:
    body = {"messages": [{"role": "user", "content": "郑州今天天气怎么样？"}]}
    data, ctx = with_web(
        body,
        "auto",
        search=lambda question: WebContext(query=question, results=fake_results),
    )
    assert ctx is not None and ctx.ok
    assert data is not body
    assert "【联网资料】" in data["messages"][0]["content"]


def test_with_web_tells_model_when_nothing_found() -> None:
    body = {"messages": [{"role": "user", "content": "郑州今天天气怎么样？"}]}
    data, ctx = with_web(body, "auto", search=lambda question: WebContext(query=question, results=[]))
    assert ctx is not None and not ctx.ok
    # 搜不到也要说一声，否则模型会拿旧知识硬答
    assert data is not body
    assert "没有拿到可用结果" in data["messages"][0]["content"]


# --------------------------------------------------------------------------
# 缓存与接口
# --------------------------------------------------------------------------
def test_research_caches_repeated_query(monkeypatch: pytest.MonkeyPatch, fake_results: list[SearchResult]) -> None:
    """缓存的是 ``research``：同一句第二次不许再打搜索引擎。

    以前这里只 patch ``bing_search``，"两个结果都不够相关"时 ``search_web`` 会
    掉进 360 兜底——于是这个测试其实在联网，机器不通 DNS 就随机失败。
    改成 patch ``search_web``，把被测面收窄到缓存本身。
    """
    calls = {"search": 0, "enrich": 0}

    def fake_search(query: str, *, limit: int = 4) -> list[SearchResult]:
        calls["search"] += 1
        return list(fake_results)

    def fake_enrich(results, *, workers: int = 4):
        calls["enrich"] += 1
        return list(results)

    monkeypatch.setattr(web_proxy, "search_web", fake_search)
    monkeypatch.setattr(web_proxy, "enrich", fake_enrich)
    web_proxy._CACHE._data.clear()

    first = research("郑州大学 2026 年招生计划")
    second = research("郑州大学 2026 年招生计划")
    assert first.ok and second.ok
    assert calls["search"] == 1
    assert second.cached is True


def test_search_endpoint_returns_results(monkeypatch: pytest.MonkeyPatch, fake_results: list[SearchResult]) -> None:
    monkeypatch.setattr(
        web_proxy,
        "research",
        lambda question, limit=4: WebContext(query=question, results=fake_results),
    )
    app = web_proxy.create_app("http://127.0.0.1:9")
    with TestClient(app) as client:
        payload = client.post("/v1/web/search", json={"query": "郑州大学"}).json()
    assert payload["query"] == "郑州大学"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["url"] == "https://www.zzu.edu.cn/"


def test_health_reports_upstream() -> None:
    app = web_proxy.create_app("http://127.0.0.1:1234")
    with TestClient(app) as client:
        payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["upstream"] == "http://127.0.0.1:1234"


# --------------------------------------------------------------------------
# 出站闸门（SSRF）与代理口令
# --------------------------------------------------------------------------
def _public_dns(host: str, port: int, **_kw):
    """把任意查询解析成一个公网地址，用来验证"放行的是正常站点"。

    用 ``8.8.8.8`` 而不是文档段 ``203.0.113.x``：Python 的 ``is_private``
    把三个 documentation 段也算作私有，用它们会测出假失败。
    """
    return [(2, 1, 6, "", ("8.8.8.8", port))]


def test_is_safe_url_blocks_internal_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """代理和 RAG 服务同机：内网地址、云元数据、非 http 协议都不许抓。"""
    monkeypatch.setattr(web_proxy.socket, "getaddrinfo", _public_dns)
    blocked = [
        "file:///etc/passwd",
        "http://127.0.0.1:8000/console",
        "http://localhost:1234/v1/models",
        "http://169.254.169.254/latest/meta-data/iam/",
        "http://10.0.0.8/",
        "http://[::1]:8000/",
        "gopher://example.com:11211/",
        "",
    ]
    for url in blocked:
        assert web_proxy.is_safe_url(url) is False, url
    assert web_proxy.is_safe_url("https://www.zzu.edu.cn/") is True


def test_is_safe_url_blocks_name_resolving_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """``kb.internal`` 这种"名字看不出问题"的域名，解析到内网一样拦掉。"""
    monkeypatch.setattr(
        web_proxy.socket,
        "getaddrinfo",
        lambda host, port, **_kw: [(2, 1, 6, "", ("127.0.0.1", port))],
    )
    assert web_proxy.is_safe_url("https://kb.internal.example/") is False


def test_fetch_page_never_touches_httpx_for_internal_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_kw):
        raise AssertionError("闸门失效：内网地址被真的请求了")

    monkeypatch.setattr(web_proxy.httpx, "get", boom)
    assert web_proxy.fetch_page("http://127.0.0.1:8000/console") == ""


def test_fetch_page_stops_at_redirect_to_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    """第一跳公网、第二跳回环：重定向必须逐跳复检，否则闸门形同虚设。"""
    hops: list[str] = []

    def fake_get(url, *_a, **_kw):
        hops.append(url)
        if url == "https://good.example/":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8000/console"})
        raise AssertionError(f"不该请求：{url}")

    monkeypatch.setattr(web_proxy.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(web_proxy.httpx, "get", fake_get)
    assert web_proxy.fetch_page("https://good.example/") == ""
    assert hops == ["https://good.example/"]


def test_proxy_requires_token_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """开了口令：没带、带错都 401；``Authorization: Bearer`` 与 ``X-Proxy-Token`` 都收；/health 保持公开。"""
    monkeypatch.setattr(
        web_proxy,
        "research",
        lambda question, limit=4: WebContext(query=question, results=[]),
    )
    app = web_proxy.create_app("http://127.0.0.1:9", token="s3cret-token")
    with TestClient(app) as client:
        body = {"query": "郑州大学"}
        assert client.post("/v1/web/search", json=body).status_code == 401
        assert client.post("/v1/web/search", json=body, headers={"X-Proxy-Token": "nope"}).status_code == 401
        assert client.post(
            "/v1/web/search", json=body, headers={"Authorization": "Bearer s3cret-token"}
        ).status_code == 200
        assert client.post("/v1/web/search", json=body, headers={"X-Proxy-Token": "s3cret-token"}).status_code == 200
        assert client.get("/health").status_code == 200


def test_proxy_without_token_stays_open_for_local_demo() -> None:
    """默认不设口令（只监听 127.0.0.1）时行为不变，本地演示不用配 api_key。"""
    app = web_proxy.create_app("http://127.0.0.1:9")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
