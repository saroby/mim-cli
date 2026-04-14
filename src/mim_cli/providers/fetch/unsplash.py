"""Unsplash 사진 API. https://unsplash.com/documentation#search-photos"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from mim_cli.net import safe_get_bytes, UnsafeURLError, DownloadTooLargeError
from mim_cli.providers import FetchProvider, FetchedMedia


class UnsplashProvider(FetchProvider):
    name = "unsplash"
    BASE_URL = "https://api.unsplash.com"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self._key = api_key or os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
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
                "UNSPLASH_ACCESS_KEY 환경변수를 설정하세요. "
                "https://unsplash.com/developers 에서 발급."
            )

        headers = {
            "Authorization": f"Client-ID {self._key}",
            "Accept-Version": "v1",
        }
        params = {
            "query": query,
            "per_page": max(1, min(limit, 30)),
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                f"{self.BASE_URL}/search/photos",
                params=params, headers=headers,
            )
            resp.raise_for_status()
            payload = resp.json()

            results = []
            for item in (payload.get("results") or [])[:limit]:
                urls = item.get("urls") or {}
                photo_url = urls.get("regular") or urls.get("full") or urls.get("small")
                if not photo_url:
                    continue
                try:
                    img_bytes, _ = safe_get_bytes(client, photo_url)
                except (httpx.HTTPError, UnsafeURLError, DownloadTooLargeError):
                    continue

                user = item.get("user") or {}
                creator = user.get("name") or user.get("username")
                # Unsplash 크레딧 규격: "Photo by <name> on Unsplash"
                attribution = f"Photo by {creator} on Unsplash" if creator else None

                results.append(FetchedMedia(
                    data=img_bytes,
                    mime_type="image/jpeg",
                    source_url=photo_url,
                    source_id=str(item.get("id", "")),
                    width=item.get("width"),
                    height=item.get("height"),
                    attribution=attribution,
                    license="Unsplash",
                    license_url="https://unsplash.com/license",
                    metadata={
                        "description": item.get("description") or item.get("alt_description"),
                        "profile_url": (user.get("links") or {}).get("html"),
                        "photo_url": (item.get("links") or {}).get("html"),
                    },
                ))

        return results
