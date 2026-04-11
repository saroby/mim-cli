"""Pixabay 이미지/비디오 API. https://pixabay.com/api/docs/"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from meme_cli.net import safe_get_bytes, UnsafeURLError, DownloadTooLargeError
from meme_cli.providers import FetchProvider, FetchedMedia


class PixabayProvider(FetchProvider):
    name = "pixabay"
    BASE_URL = "https://pixabay.com/api"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self._key = api_key or os.environ.get("PIXABAY_API_KEY", "").strip()
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
                "PIXABAY_API_KEY 환경변수를 설정하세요. "
                "https://pixabay.com/api/docs/ 에서 발급."
            )

        params = {
            "key": self._key,
            "q": query,
            "per_page": max(3, min(limit, 200)),
            "safesearch": "true",
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{self.BASE_URL}/", params=params)
            resp.raise_for_status()
            payload = resp.json()

            results = []
            for item in (payload.get("hits") or [])[:limit]:
                image_url = item.get("largeImageURL") or item.get("webformatURL")
                if not image_url:
                    continue
                try:
                    img_bytes, _ = safe_get_bytes(client, image_url)
                except (httpx.HTTPError, UnsafeURLError, DownloadTooLargeError):
                    continue

                creator = item.get("user")
                attribution = f"Image by {creator} on Pixabay" if creator else None

                results.append(FetchedMedia(
                    data=img_bytes,
                    mime_type="image/jpeg",
                    source_url=image_url,
                    source_id=str(item.get("id", "")),
                    width=item.get("imageWidth"),
                    height=item.get("imageHeight"),
                    attribution=attribution,
                    license="Pixabay Content License",
                    license_url="https://pixabay.com/service/license-summary/",
                    metadata={
                        "tags": item.get("tags"),
                        "page_url": item.get("pageURL"),
                        "user_id": item.get("user_id"),
                    },
                ))

        return results
