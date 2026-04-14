"""Pexels 사진/비디오 API. https://www.pexels.com/api/documentation/"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import httpx

from meme_cli.net import safe_get_bytes, UnsafeURLError, DownloadTooLargeError
from meme_cli.providers import FetchProvider, FetchedMedia


class PexelsProvider(FetchProvider):
    name = "pexels"
    BASE_URL = "https://api.pexels.com/v1"
    VIDEOS_URL = "https://api.pexels.com/videos"

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

        params = {"query": query, "per_page": max(1, min(limit, 80))}
        headers = {"Authorization": self._key}

        if media_type == "video":
            return self._execute(
                f"{self.VIDEOS_URL}/search", params, headers,
                items_key="videos", limit=limit, extract=self._extract_video,
            )
        return self._execute(
            f"{self.BASE_URL}/search", params, headers,
            items_key="photos", limit=limit, extract=self._extract_photo,
        )

    def _execute(
        self,
        endpoint: str,
        params: dict,
        headers: dict,
        *,
        items_key: str,
        limit: int,
        extract: Callable[[dict], list[dict]],
    ) -> list[FetchedMedia]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(endpoint, params=params, headers=headers)
            resp.raise_for_status()
            results: list[FetchedMedia] = []
            for item in (resp.json().get(items_key) or [])[:limit]:
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
    def _extract_photo(item: dict) -> list[dict[str, Any]]:
        sources = item.get("src") or {}
        urls = [sources.get("large"), sources.get("original"), sources.get("medium")]
        creator = item.get("photographer")
        base: dict[str, Any] = {
            "mime_type": "image/jpeg",
            "source_id": str(item.get("id", "")),
            "width": item.get("width"),
            "height": item.get("height"),
            "attribution": f"Photo by {creator} on Pexels" if creator else None,
            "license": "Pexels",
            "license_url": "https://www.pexels.com/license/",
            "metadata": {
                "alt": item.get("alt"),
                "photographer_url": item.get("photographer_url"),
                "photo_url": item.get("url"),
            },
        }
        return [{"source_url": u, **base} for u in urls if u]

    @staticmethod
    def _extract_video(item: dict) -> list[dict[str, Any]]:
        variants = _rank_pexels_video_files(item.get("video_files") or [])
        if not variants:
            return []
        user = item.get("user") or {}
        creator = user.get("name")
        # 사진 id와 네임스페이스 분리: (source_provider, source_id) dedupe 키가 사진·비디오 교차 alias되는 것 방지.
        source_id = f"video:{item.get('id', '')}"
        candidates = []
        for v in variants:
            candidates.append({
                "source_url": v["link"],
                "mime_type": "video/mp4",
                "source_id": source_id,
                "width": v.get("width") or item.get("width"),
                "height": v.get("height") or item.get("height"),
                "attribution": f"Video by {creator} on Pexels" if creator else None,
                "license": "Pexels",
                "license_url": "https://www.pexels.com/license/",
                "metadata": {
                    "duration": item.get("duration"),
                    "user_url": user.get("url"),
                    "video_page_url": item.get("url"),
                    "quality": v.get("quality"),
                },
            })
        return candidates


def _rank_pexels_video_files(files: list[dict]) -> list[dict]:
    # mp4만: 다른 포맷 fallback 시 mime이 video/mp4로 잘못 라벨링됨.
    # sd 우선: net.py 50MB 상한에 들어올 확률이 가장 높음. 실패 시 다음 품질로 fallback.
    mp4s = [f for f in files if (f.get("file_type") or "").lower() == "video/mp4" and f.get("link")]
    if not mp4s:
        return []
    by_quality = {"sd": [], "hd": [], "other": []}
    for f in mp4s:
        q = (f.get("quality") or "").lower()
        (by_quality[q] if q in by_quality else by_quality["other"]).append(f)
    return by_quality["sd"] + by_quality["hd"] + by_quality["other"]
