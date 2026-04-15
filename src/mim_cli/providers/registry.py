"""프로바이더 레지스트리 — cli.py / server.py 공유."""

from __future__ import annotations

from mim_cli.providers import FetchProvider, ImageProvider

GEN_PROVIDERS: dict[str, type[ImageProvider]] = {}
FETCH_PROVIDERS: dict[str, type[FetchProvider]] = {}


def register_providers() -> None:
    from mim_cli.providers.gemini import GeminiProvider
    from mim_cli.providers.leonardo import LeonardoProvider
    from mim_cli.providers.replicate import ReplicateProvider
    from mim_cli.providers.fetch.giphy import GiphyProvider
    from mim_cli.providers.fetch.openverse import OpenverseProvider
    from mim_cli.providers.fetch.pexels import PexelsProvider
    from mim_cli.providers.fetch.pixabay import PixabayProvider
    from mim_cli.providers.fetch.reddit import RedditProvider
    from mim_cli.providers.fetch.unsplash import UnsplashProvider

    GEN_PROVIDERS.update({
        "gemini": GeminiProvider,
        "replicate": ReplicateProvider,
        "leonardo": LeonardoProvider,
    })
    FETCH_PROVIDERS.update({
        "giphy": GiphyProvider,
        "reddit": RedditProvider,
        "unsplash": UnsplashProvider,
        "pexels": PexelsProvider,
        "pixabay": PixabayProvider,
        "openverse": OpenverseProvider,
    })


register_providers()


def suffix_from_mime(mime: str) -> str:
    """MIME → 파일 확장자."""
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
    }.get(mime.lower(), ".bin")


def media_type_from_mime(mime: str) -> str:
    """MIME → media_type."""
    m = mime.lower()
    if m.startswith("video/"):
        return "video"
    if m == "image/gif":
        return "gif"
    return "image"


def media_type_from_suffix(suffix: str) -> str:
    """확장자 → media_type."""
    s = suffix.lower()
    if s in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        return "video"
    if s == ".gif":
        return "gif"
    return "image"
