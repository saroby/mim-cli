"""net.py의 SSRF/다운로드-크기 방어 단위 테스트."""

from __future__ import annotations

import httpx
import pytest

from meme_cli.net import (
    DownloadTooLargeError,
    UnsafeURLError,
    safe_get_bytes,
    validate_url,
)


def test_rejects_loopback_literal():
    with pytest.raises(UnsafeURLError):
        validate_url("http://127.0.0.1/x")


def test_rejects_private_literal():
    with pytest.raises(UnsafeURLError):
        validate_url("http://192.168.0.1/x")


def test_rejects_link_local():
    with pytest.raises(UnsafeURLError):
        validate_url("http://169.254.169.254/latest/meta-data/")


def test_rejects_file_scheme():
    with pytest.raises(UnsafeURLError):
        validate_url("file:///etc/passwd")


def test_rejects_localhost_hostname():
    # localhost는 DNS에서 127.0.0.1로 해석됨
    with pytest.raises(UnsafeURLError):
        validate_url("http://localhost/x")


def test_size_cap_enforced_via_stream():
    big = b"x" * 2048

    def handler(request: httpx.Request) -> httpx.Response:
        # Content-Length 헤더 없이 반환해 스트림 누적 가드 검증
        return httpx.Response(200, content=big)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(DownloadTooLargeError):
            safe_get_bytes(
                client, "https://cdn.example.com/big",
                max_bytes=1024,
            )


def test_normal_download_works():
    payload = b"\xff\xd8\xff hello"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=payload,
            headers={"content-type": "image/jpeg"},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        data, ct = safe_get_bytes(client, "https://cdn.example.com/ok")
    assert data == payload
    assert ct == "image/jpeg"
