"""Reddit 미디어 검색. 기본 무인증, OAuth 선택 (더 높은 레이트 리밋).

무인증 엔드포인트:
  GET https://www.reddit.com/search.json?q=<query>&limit=N
  GET https://www.reddit.com/r/<sub>/search.json?q=<query>&restrict_sr=on&limit=N

OAuth 엔드포인트 (REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET 있을 때):
  POST https://www.reddit.com/api/v1/access_token → Bearer 토큰
  GET  https://oauth.reddit.com/search (동일한 쿼리 구조)

Reddit은 고유한 User-Agent를 요구하므로 반드시 지정.
"""

from __future__ import annotations

import html
import os
from typing import Any, Optional

import httpx

from meme_cli.net import safe_get_bytes, UnsafeURLError, DownloadTooLargeError
from meme_cli.providers import FetchProvider, FetchedMedia


DEFAULT_USER_AGENT = "meme-cli/0.1 (+https://github.com/)"


class RedditProvider(FetchProvider):
    name = "reddit"
    PUBLIC_URL = "https://www.reddit.com"
    OAUTH_URL = "https://oauth.reddit.com"
    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

    def __init__(
        self,
        timeout: float = 120.0,
        subreddit: Optional[str] = None,
    ):
        self._timeout = timeout
        self._subreddit = subreddit or os.environ.get("REDDIT_DEFAULT_SUBREDDIT", "").strip() or None
        self._user_agent = os.environ.get("REDDIT_USER_AGENT", "").strip() or DEFAULT_USER_AGENT
        self._client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
        self._client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
        self._token: Optional[str] = None

    def check_auth(self) -> bool:
        # 무인증도 동작. OAuth 키가 있으면 더 높은 한도.
        return True

    def _get_oauth_token(self, client: httpx.Client) -> Optional[str]:
        if self._token:
            return self._token
        if not (self._client_id and self._client_secret):
            return None
        resp = client.post(
            self.TOKEN_URL,
            auth=(self._client_id, self._client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": self._user_agent},
        )
        resp.raise_for_status()
        self._token = resp.json().get("access_token")
        return self._token

    def search(
        self,
        query: str,
        limit: int = 1,
        media_type: Optional[str] = None,
    ) -> list[FetchedMedia]:
        headers = {"User-Agent": self._user_agent}

        # 과잉 수집 후 클라이언트에서 필터 (text-only 게시물 제거)
        params = {
            "q": query,
            "limit": max(1, min(limit * 4, 100)),
            "sort": "relevance",
            "type": "link",
            "raw_json": 1,  # HTML 이스케이프 제거 (preview URL 디코딩)
        }
        subreddit = self._subreddit
        if subreddit:
            path = f"/r/{subreddit}/search.json"
            params["restrict_sr"] = "on"
        else:
            path = "/search.json"

        with httpx.Client(timeout=self._timeout) as client:
            # OAuth 사용 가능하면 활용
            token = self._get_oauth_token(client)
            if token:
                base = self.OAUTH_URL
                headers["Authorization"] = f"Bearer {token}"
                # oauth 엔드포인트는 .json 접미사 없음
                url = f"{base}{path.replace('.json', '')}"
            else:
                url = f"{self.PUBLIC_URL}{path}"

            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

            results: list[FetchedMedia] = []
            for child in (payload.get("data", {}).get("children") or []):
                if len(results) >= limit:
                    break
                post = child.get("data") or {}
                media_url, mime, w, h = _extract_media(post)
                if not media_url or not _matches_media_type(mime, media_type):
                    continue

                try:
                    media_bytes, _ = safe_get_bytes(
                        client, media_url,
                        headers={"User-Agent": self._user_agent},
                    )
                except (httpx.HTTPError, UnsafeURLError, DownloadTooLargeError):
                    continue

                permalink = post.get("permalink") or ""
                results.append(FetchedMedia(
                    data=media_bytes,
                    mime_type=mime,
                    source_url=media_url,
                    source_id=str(post.get("id", "")),
                    width=w,
                    height=h,
                    attribution=f"u/{post.get('author')}" if post.get("author") else None,
                    license="Reddit",
                    license_url="https://www.redditinc.com/policies/user-agreement",
                    metadata={
                        "title": post.get("title"),
                        "subreddit": post.get("subreddit"),
                        "permalink": f"https://www.reddit.com{permalink}" if permalink else None,
                        "score": post.get("score"),
                        "over_18": post.get("over_18"),
                    },
                ))

        return results


def _extract_media(post: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    """Reddit 포스트에서 (media_url, mime, width, height) 추출. 미디어 없으면 (None,*)."""
    url = (post.get("url") or "").lower()
    preview_w, preview_h = _preview_dims(post)

    # 1) 직접 이미지/GIF URL
    if url.endswith((".jpg", ".jpeg")):
        return post["url"], "image/jpeg", preview_w, preview_h
    if url.endswith(".png"):
        return post["url"], "image/png", preview_w, preview_h
    if url.endswith(".webp"):
        return post["url"], "image/webp", preview_w, preview_h
    if url.endswith(".gif"):
        return post["url"], "image/gif", preview_w, preview_h

    # 2) preview.variants (GIF/MP4 선호)
    preview_images = (post.get("preview") or {}).get("images") or []
    if preview_images:
        variants = preview_images[0].get("variants") or {}
        gif = (variants.get("gif") or {}).get("source")
        if gif and gif.get("url"):
            return html.unescape(gif["url"]), "image/gif", gif.get("width"), gif.get("height")
        mp4 = (variants.get("mp4") or {}).get("source")
        if mp4 and mp4.get("url"):
            return html.unescape(mp4["url"]), "video/mp4", mp4.get("width"), mp4.get("height")
        source = preview_images[0].get("source")
        if source and source.get("url"):
            return (
                html.unescape(source["url"]),
                "image/jpeg",
                source.get("width"),
                source.get("height"),
            )

    # 3) v.redd.it (네이티브 Reddit 비디오)
    reddit_video = (post.get("media") or {}).get("reddit_video")
    if reddit_video and reddit_video.get("fallback_url"):
        return (
            reddit_video["fallback_url"],
            "video/mp4",
            reddit_video.get("width"),
            reddit_video.get("height"),
        )

    return None, None, None, None


def _preview_dims(post: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    images = (post.get("preview") or {}).get("images") or []
    if not images:
        return None, None
    source = images[0].get("source") or {}
    return source.get("width"), source.get("height")


def _matches_media_type(mime: Optional[str], wanted: Optional[str]) -> bool:
    if not mime:
        return False
    if not wanted:
        return True
    if wanted == "gif":
        return "gif" in mime
    if wanted == "image":
        return mime.startswith("image/") and "gif" not in mime
    if wanted == "video":
        return mime.startswith("video/")
    return True
