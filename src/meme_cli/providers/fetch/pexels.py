"""Pexels 사진/비디오 API. https://www.pexels.com/api/documentation/"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from meme_cli.net import safe_get_bytes, UnsafeURLError, DownloadTooLargeError
from meme_cli.providers import FetchProvider, FetchedMedia


class PexelsProvider(FetchProvider):
    name = "pexels"
    BASE_URL = "https://api.pexels.com/v1"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self._key = api_key or os.environ.get("PEXELS_API_KEY", "").strip()
        self._timeout = timeout

    def check_auth(self) -> bool:
        return bool(self._key)

    def search(
        self,
        query: str,
        limit: int = 1,
        media_type: Optional[str] = None,
    ) -> list[FetchedMedia]:
        if not self._key:
            raise RuntimeError(
                "PEXELS_API_KEY 환경변수를 설정하세요. "
                "https://www.pexels.com/api/new/ 에서 발급."
            )

        headers = {"Authorization": self._key}
        params = {
            "query": query,
            "per_page": max(1, min(limit, 80)),
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                f"{self.BASE_URL}/search", params=params, headers=headers,
            )
            resp.raise_for_status()
            payload = resp.json()

            results = []
            for item in (payload.get("photos") or [])[:limit]:
                sources = item.get("src") or {}
                photo_url = sources.get("large") or sources.get("original") or sources.get("medium")
                if not photo_url:
                    continue
                try:
                    img_bytes, _ = safe_get_bytes(client, photo_url)
                except (httpx.HTTPError, UnsafeURLError, DownloadTooLargeError):
                    continue

                creator = item.get("photographer")
                attribution = f"Photo by {creator} on Pexels" if creator else None

                results.append(FetchedMedia(
                    data=img_bytes,
                    mime_type="image/jpeg",
                    source_url=photo_url,
                    source_id=str(item.get("id", "")),
                    width=item.get("width"),
                    height=item.get("height"),
                    attribution=attribution,
                    license="Pexels",
                    license_url="https://www.pexels.com/license/",
                    metadata={
                        "alt": item.get("alt"),
                        "photographer_url": item.get("photographer_url"),
                        "photo_url": item.get("url"),
                    },
                ))

        return results
