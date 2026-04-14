"""Pixabay 이미지/비디오 API. https://pixabay.com/api/docs/"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import httpx

from mim_cli.net import safe_get_bytes, UnsafeURLError, DownloadTooLargeError
from mim_cli.providers import FetchProvider, FetchedMedia


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

        # Pixabay API는 per_page 최소값 3을 요구.
        params = {
            "key": self._key,
            "q": query,
            "per_page": max(3, min(limit, 200)),
            "safesearch": "true",
        }

        if media_type == "video":
            return self._execute(
                f"{self.BASE_URL}/videos/", params,
                limit=limit, extract=self._extract_video,
            )
        return self._execute(
            f"{self.BASE_URL}/", params,
            limit=limit, extract=self._extract_image,
        )

    def _execute(
        self,
        endpoint: str,
        params: dict,
        *,
        limit: int,
        extract: Callable[[dict], list[dict]],
    ) -> list[FetchedMedia]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(endpoint, params=params)
            resp.raise_for_status()
            results: list[FetchedMedia] = []
            for item in (resp.json().get("hits") or [])[:limit]:
                for candidate in extract(item):
                    url = candidate["source_url"]
                    try:
                        data, _ = safe_get_bytes(client, url)
                    except (httpx.HTTPError, UnsafeURLError, DownloadTooLargeError):
                        # 다음 rendition으로 fallback. 50MB 상한, CDN 장애 대응.
                        continue
                    results.append(FetchedMedia(data=data, **candidate))
                    break
            return results

    @staticmethod
    def _extract_image(item: dict) -> list[dict[str, Any]]:
        urls = [item.get("largeImageURL"), item.get("webformatURL")]
        creator = item.get("user")
        base: dict[str, Any] = {
            "mime_type": "image/jpeg",
            "source_id": str(item.get("id", "")),
            "width": item.get("imageWidth"),
            "height": item.get("imageHeight"),
            "attribution": f"Image by {creator} on Pixabay" if creator else None,
            "license": "Pixabay Content License",
            "license_url": "https://pixabay.com/service/license-summary/",
            "metadata": {
                "tags": item.get("tags"),
                "page_url": item.get("pageURL"),
                "user_id": item.get("user_id"),
            },
        }
        return [{"source_url": u, **base} for u in urls if u]

    @staticmethod
    def _extract_video(item: dict) -> list[dict[str, Any]]:
        variants = _rank_pixabay_videos(item.get("videos") or {})
        if not variants:
            return []
        creator = item.get("user")
        # 이미지 id와 네임스페이스 분리: dedupe 키가 이미지·비디오 교차 alias되는 것 방지.
        source_id = f"video:{item.get('id', '')}"
        candidates = []
        for v in variants:
            candidates.append({
                "source_url": v["url"],
                "mime_type": "video/mp4",
                "source_id": source_id,
                "width": v.get("width"),
                "height": v.get("height"),
                "attribution": f"Video by {creator} on Pixabay" if creator else None,
                "license": "Pixabay Content License",
                "license_url": "https://pixabay.com/service/license-summary/",
                "metadata": {
                    "tags": item.get("tags"),
                    "page_url": item.get("pageURL"),
                    "user_id": item.get("user_id"),
                    "duration": item.get("duration"),
                    "quality": v.get("_quality"),
                },
            })
        return candidates


def _rank_pixabay_videos(variants: dict) -> list[dict]:
    # small(960p) 우선 — 품질/크기 균형. large(1080p)는 50MB 상한을 넘기 쉬워 최후.
    # 실패 시 다음 variant로 fallback.
    ranked = []
    for key in ("small", "medium", "tiny", "large"):
        v = variants.get(key)
        if v and v.get("url"):
            ranked.append({**v, "_quality": key})
    return ranked
