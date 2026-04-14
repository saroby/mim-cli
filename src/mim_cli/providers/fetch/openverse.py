"""Openverse — CC 라이선스 이미지/오디오. API 키 불필요.

https://api.openverse.org/v1/
"""

from __future__ import annotations

from typing import Optional

import httpx

from mim_cli.net import safe_get_bytes, UnsafeURLError, DownloadTooLargeError
from mim_cli.providers import FetchProvider, FetchedMedia


class OpenverseProvider(FetchProvider):
    """Openverse CC 라이선스 미디어 검색."""

    name = "openverse"
    BASE_URL = "https://api.openverse.org/v1"

    def __init__(self, timeout: float = 120.0):
        self._timeout = timeout

    def check_auth(self) -> bool:
        return True  # 항상 사용 가능

    def search(
        self,
        query: str,
        limit: int = 1,
        media_type: Optional[str] = None,
    ) -> list[FetchedMedia]:
        # Openverse는 이미지/오디오 별도 엔드포인트. media_type=image만 이번 범위.
        params = {"q": query, "page_size": max(1, min(limit, 20))}

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{self.BASE_URL}/images/", params=params)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in (data.get("results") or [])[:limit]:
                img_url = item.get("url") or item.get("thumbnail")
                if not img_url:
                    continue
                try:
                    img_bytes, ct = safe_get_bytes(client, img_url)
                except (httpx.HTTPError, UnsafeURLError, DownloadTooLargeError):
                    continue

                mime = ct or "image/jpeg"
                results.append(FetchedMedia(
                    data=img_bytes,
                    mime_type=mime,
                    source_url=img_url,
                    source_id=str(item.get("id", "")),
                    width=item.get("width"),
                    height=item.get("height"),
                    attribution=item.get("creator"),
                    license=item.get("license"),
                    license_url=item.get("license_url"),
                    metadata={
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "foreign_landing_url": item.get("foreign_landing_url"),
                    },
                ))

        return results
