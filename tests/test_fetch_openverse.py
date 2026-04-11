"""Openverse fetch 프로바이더 mock 테스트. respx 기반."""

import httpx
import pytest
import respx

from meme_cli.providers.fetch.openverse import OpenverseProvider


@pytest.fixture
def provider():
    return OpenverseProvider()


@respx.mock
def test_search_returns_fetched_media(provider):
    respx.get("https://api.openverse.org/v1/images/").mock(
        return_value=httpx.Response(200, json={
            "results": [{
                "id": "abc-123",
                "title": "Cute cat",
                "url": "https://cdn.example.com/cat.jpg",
                "creator": "Jane Doe",
                "license": "cc0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "width": 800,
                "height": 600,
                "source": "flickr",
                "foreign_landing_url": "https://flickr.com/photos/jane/123",
            }],
        })
    )
    respx.get("https://cdn.example.com/cat.jpg").mock(
        return_value=httpx.Response(
            200,
            content=b"\xff\xd8\xff binary jpeg bytes",
            headers={"content-type": "image/jpeg"},
        )
    )

    results = provider.search("cat", limit=1)

    assert len(results) == 1
    r = results[0]
    assert r.source_id == "abc-123"
    assert r.source_url == "https://cdn.example.com/cat.jpg"
    assert r.mime_type == "image/jpeg"
    assert r.data == b"\xff\xd8\xff binary jpeg bytes"
    assert r.width == 800
    assert r.height == 600
    assert r.attribution == "Jane Doe"
    assert r.license == "cc0"
    assert r.license_url.startswith("https://creativecommons.org")
    assert r.metadata["title"] == "Cute cat"
    assert r.metadata["source"] == "flickr"


@respx.mock
def test_search_empty_result(provider):
    respx.get("https://api.openverse.org/v1/images/").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    assert provider.search("nothing", limit=5) == []


@respx.mock
def test_search_limit_respected(provider):
    respx.get("https://api.openverse.org/v1/images/").mock(
        return_value=httpx.Response(200, json={
            "results": [
                {
                    "id": str(i), "url": f"https://cdn.example.com/{i}.jpg",
                    "license": "cc-by",
                } for i in range(10)
            ],
        })
    )
    for i in range(10):
        respx.get(f"https://cdn.example.com/{i}.jpg").mock(
            return_value=httpx.Response(200, content=b"img", headers={"content-type": "image/jpeg"})
        )
    results = provider.search("x", limit=3)
    assert len(results) == 3
    assert results[0].source_id == "0"
    assert results[2].source_id == "2"


@respx.mock
def test_search_skips_item_with_failed_download(provider):
    respx.get("https://api.openverse.org/v1/images/").mock(
        return_value=httpx.Response(200, json={
            "results": [
                {"id": "ok-1", "url": "https://cdn.example.com/a.jpg", "license": "cc0"},
                {"id": "bad-1", "url": "https://cdn.example.com/b.jpg", "license": "cc0"},
            ],
        })
    )
    respx.get("https://cdn.example.com/a.jpg").mock(
        return_value=httpx.Response(200, content=b"ok", headers={"content-type": "image/jpeg"})
    )
    respx.get("https://cdn.example.com/b.jpg").mock(
        return_value=httpx.Response(500)  # 다운로드 실패
    )
    results = provider.search("x", limit=5)
    assert len(results) == 1
    assert results[0].source_id == "ok-1"


def test_check_auth_always_true():
    # API 키 없어도 동작
    assert OpenverseProvider().check_auth() is True
