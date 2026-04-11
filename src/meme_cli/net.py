"""안전한 HTTP 다운로드 유틸리티.

보안 목적:
  - SSRF 방지: URL 호스트가 사설/루프백/링크로컬 IP로 해석되면 거부.
  - 메모리 보호: 응답 바이트 크기에 상한 강제 (기본 50MB).
  - 스킴 제한: http/https만 허용.

fetch 프로바이더들이 외부 API 응답에 들어있는 URL(공격자가 심을 수 있는 값)로
미디어를 다운로드할 때 공통으로 사용.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx


# 기본 미디어 다운로드 상한. 50MB면 대부분의 밈(이미지/GIF/짧은 MP4)을 수용.
DEFAULT_MAX_BYTES = 50 * 1024 * 1024


class UnsafeURLError(ValueError):
    """SSRF 방지: 거부된 URL."""


class DownloadTooLargeError(ValueError):
    """응답 크기가 상한을 초과."""


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def validate_url(url: str) -> None:
    """URL 스킴/호스트 검사. 거부 사유가 있으면 UnsafeURLError raise."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(f"허용되지 않은 스킴: {parsed.scheme}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("호스트 없음")

    # 리터럴 IP 즉시 체크
    try:
        if _is_private_ip(host):
            raise UnsafeURLError(f"사설/루프백 주소 거부: {host}")
    except ValueError:
        pass

    # DNS 해석 후 모든 주소가 public인지 확인.
    # 해석 실패(gaierror)는 SSRF가 아니고 네트워크 장애이므로 fall-open —
    # 이후 httpx.stream이 ConnectError를 내는 경로에 맡김.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return
    for info in infos:
        ip = info[4][0]
        if _is_private_ip(ip):
            raise UnsafeURLError(f"사설/루프백 주소로 해석됨: {host} → {ip}")


def safe_get_bytes(
    client: httpx.Client,
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    headers: dict | None = None,
) -> tuple[bytes, str]:
    """URL을 검증한 뒤 스트리밍으로 다운로드. (bytes, content_type) 반환.

    Content-Length 힌트가 max_bytes를 초과하면 즉시 거부.
    스트림 중 누적 바이트가 상한을 넘으면 DownloadTooLargeError.
    """
    validate_url(url)
    with client.stream("GET", url, headers=headers) as resp:
        resp.raise_for_status()

        length = resp.headers.get("content-length")
        if length:
            try:
                if int(length) > max_bytes:
                    raise DownloadTooLargeError(
                        f"응답 크기 {length} > 상한 {max_bytes}"
                    )
            except ValueError:
                pass

        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise DownloadTooLargeError(
                    f"응답 크기가 상한 {max_bytes} 초과"
                )
            chunks.append(chunk)

        content_type = resp.headers.get("content-type", "").split(";")[0].strip()
        return b"".join(chunks), content_type
