"""Giphy GIF API. https://developers.giphy.com/docs/api/endpoint/#search"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from mim_cli.net import safe_get_bytes, UnsafeURLError, DownloadTooLargeError
from mim_cli.providers import FetchProvider, FetchedMedia


class GiphyProvider(FetchProvider):
    name = "giphy"
    BASE_URL = "https://api.giphy.com/v1/gifs"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self._key = api_key or os.environ.get("GIPHY_API_KEY", "").strip()
        self._timeout = timeout

    def check_auth(self) -> bool:
        return bool(self._key)

    def search(
        self,
        query: str,
        limit: int = 1,
        media_type: Optional[str] = None,
    ) -> list[FetchedMedia]:
        # media_type이 명시적으로 image면 Giphy는 지원 안 함. 무시하고 GIF 반환.
        if not self._key:
            raise RuntimeError(
                "GIPHY_API_KEY 환경변수를 설정하세요. "
                "https://developers.giphy.com/dashboard/ 에서 발급."
            )

        params = {
            "api_key": self._key,
            "q": query,
            "limit": max(1, min(limit, 50)),
            "rating": "pg-13",
            "lang": "ko",
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{self.BASE_URL}/search", params=params)
            resp.raise_for_status()
            payload = resp.json()

            results = []
            for item in (payload.get("data") or [])[:limit]:
                images = item.get("images") or {}
                # 원본 품질 GIF 우선, 없으면 downsized
                variant = images.get("original") or images.get("downsized") or {}
                gif_url = variant.get("url")
                if not gif_url:
                    continue
                try:
                    gif_bytes, _ = safe_get_bytes(client, gif_url)
                except (httpx.HTTPError, UnsafeURLError, DownloadTooLargeError):
                    continue

                results.append(FetchedMedia(
                    data=gif_bytes,
                    mime_type="image/gif",
                    source_url=gif_url,
                    source_id=str(item.get("id", "")),
                    width=_safe_int(variant.get("width")),
                    height=_safe_int(variant.get("height")),
                    attribution=(item.get("user") or {}).get("display_name") or item.get("username"),
                    license="Giphy",
                    license_url="https://giphy.com/terms",
                    metadata={
                        "title": item.get("title"),
                        "slug": item.get("slug"),
                        "bitly_url": item.get("bitly_url"),
                    },
                ))

        return results


def _safe_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
