"""Pexels/Pixabay 비디오 경로: variant fallback + source_id 네임스페이싱 검증."""

import httpx
import respx

from meme_cli.providers.fetch.pexels import PexelsProvider
from meme_cli.providers.fetch.pixabay import PixabayProvider


# --- Pexels ---


def _pexels_video_payload(video_id: int, files: list[dict]) -> dict:
    return {
        "videos": [{
            "id": video_id,
            "width": 1920,
            "height": 1080,
            "duration": 10,
            "url": f"https://www.pexels.com/video/{video_id}/",
            "user": {"name": "creator", "url": "https://www.pexels.com/@creator"},
            "video_files": files,
        }]
    }


@respx.mock
def test_pexels_video_falls_back_when_sd_too_large():
    files = [
        {"link": "https://cdn.pexels.com/sd.mp4", "file_type": "video/mp4", "quality": "sd", "width": 960, "height": 540},
        {"link": "https://cdn.pexels.com/hd.mp4", "file_type": "video/mp4", "quality": "hd", "width": 1920, "height": 1080},
    ]
    respx.get("https://api.pexels.com/videos/search").mock(
        return_value=httpx.Response(200, json=_pexels_video_payload(123, files))
    )
    # sd 다운로드 실패 (CDN 5xx) → hd로 fallback
    respx.get("https://cdn.pexels.com/sd.mp4").mock(return_value=httpx.Response(503))
    respx.get("https://cdn.pexels.com/hd.mp4").mock(
        return_value=httpx.Response(200, content=b"MP4DATA", headers={"content-type": "video/mp4"})
    )

    results = PexelsProvider(api_key="k").search("cat", limit=1, media_type="video")
    assert len(results) == 1
    assert results[0].source_url == "https://cdn.pexels.com/hd.mp4"
    assert results[0].mime_type == "video/mp4"
    assert results[0].data == b"MP4DATA"


@respx.mock
def test_pexels_video_source_id_namespaced():
    files = [{"link": "https://cdn.pexels.com/sd.mp4", "file_type": "video/mp4", "quality": "sd"}]
    respx.get("https://api.pexels.com/videos/search").mock(
        return_value=httpx.Response(200, json=_pexels_video_payload(999, files))
    )
    respx.get("https://cdn.pexels.com/sd.mp4").mock(
        return_value=httpx.Response(200, content=b"MP4", headers={"content-type": "video/mp4"})
    )

    results = PexelsProvider(api_key="k").search("x", limit=1, media_type="video")
    assert results[0].source_id == "video:999"


@respx.mock
def test_pexels_video_skips_when_all_variants_fail():
    files = [
        {"link": "https://cdn.pexels.com/sd.mp4", "file_type": "video/mp4", "quality": "sd"},
        {"link": "https://cdn.pexels.com/hd.mp4", "file_type": "video/mp4", "quality": "hd"},
    ]
    respx.get("https://api.pexels.com/videos/search").mock(
        return_value=httpx.Response(200, json=_pexels_video_payload(5, files))
    )
    respx.get("https://cdn.pexels.com/sd.mp4").mock(return_value=httpx.Response(500))
    respx.get("https://cdn.pexels.com/hd.mp4").mock(return_value=httpx.Response(500))

    results = PexelsProvider(api_key="k").search("x", limit=1, media_type="video")
    assert results == []


@respx.mock
def test_pexels_video_skips_non_mp4_variants():
    # webm만 있으면 빈 결과 (mp4로 잘못 라벨링 방지)
    files = [{"link": "https://cdn.pexels.com/v.webm", "file_type": "video/webm", "quality": "sd"}]
    respx.get("https://api.pexels.com/videos/search").mock(
        return_value=httpx.Response(200, json=_pexels_video_payload(7, files))
    )
    results = PexelsProvider(api_key="k").search("x", limit=1, media_type="video")
    assert results == []


# --- Pixabay ---


def _pixabay_video_payload(video_id: int, videos: dict) -> dict:
    return {
        "hits": [{
            "id": video_id,
            "pageURL": f"https://pixabay.com/videos/id-{video_id}/",
            "tags": "cat",
            "duration": 10,
            "user": "creator",
            "user_id": 42,
            "videos": videos,
        }]
    }


@respx.mock
def test_pixabay_video_falls_back_when_small_fails():
    videos = {
        "small": {"url": "https://cdn.pixabay.com/small.mp4", "width": 960, "height": 540, "size": 1000},
        "medium": {"url": "https://cdn.pixabay.com/medium.mp4", "width": 1280, "height": 720, "size": 2000},
    }
    respx.get("https://pixabay.com/api/videos/").mock(
        return_value=httpx.Response(200, json=_pixabay_video_payload(321, videos))
    )
    respx.get("https://cdn.pixabay.com/small.mp4").mock(return_value=httpx.Response(503))
    respx.get("https://cdn.pixabay.com/medium.mp4").mock(
        return_value=httpx.Response(200, content=b"MP4DATA", headers={"content-type": "video/mp4"})
    )

    results = PixabayProvider(api_key="k").search("cat", limit=1, media_type="video")
    assert len(results) == 1
    assert results[0].source_url == "https://cdn.pixabay.com/medium.mp4"
    assert results[0].width == 1280
    assert results[0].metadata["quality"] == "medium"


@respx.mock
def test_pixabay_video_source_id_namespaced():
    videos = {"small": {"url": "https://cdn.pixabay.com/small.mp4", "width": 960, "height": 540}}
    respx.get("https://pixabay.com/api/videos/").mock(
        return_value=httpx.Response(200, json=_pixabay_video_payload(777, videos))
    )
    respx.get("https://cdn.pixabay.com/small.mp4").mock(
        return_value=httpx.Response(200, content=b"MP4", headers={"content-type": "video/mp4"})
    )

    results = PixabayProvider(api_key="k").search("x", limit=1, media_type="video")
    assert results[0].source_id == "video:777"


@respx.mock
def test_pixabay_video_skips_when_all_variants_fail():
    videos = {
        "small": {"url": "https://cdn.pixabay.com/small.mp4", "width": 960},
        "medium": {"url": "https://cdn.pixabay.com/medium.mp4", "width": 1280},
        "tiny": {"url": "https://cdn.pixabay.com/tiny.mp4", "width": 640},
        "large": {"url": "https://cdn.pixabay.com/large.mp4", "width": 1920},
    }
    respx.get("https://pixabay.com/api/videos/").mock(
        return_value=httpx.Response(200, json=_pixabay_video_payload(1, videos))
    )
    for size in ("small", "medium", "tiny", "large"):
        respx.get(f"https://cdn.pixabay.com/{size}.mp4").mock(return_value=httpx.Response(500))

    results = PixabayProvider(api_key="k").search("x", limit=1, media_type="video")
    assert results == []
