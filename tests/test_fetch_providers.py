"""Giphy/Unsplash/Pexels mock 테스트. 공통 패턴: API 응답 → 이미지 다운로드 → FetchedMedia."""

import httpx
import pytest
import respx

from mim_cli.providers.fetch.giphy import GiphyProvider
from mim_cli.providers.fetch.pexels import PexelsProvider
from mim_cli.providers.fetch.pixabay import PixabayProvider
from mim_cli.providers.fetch.unsplash import UnsplashProvider


# ────────── Giphy ──────────


@respx.mock
def test_giphy_search():
    respx.get("https://api.giphy.com/v1/gifs/search").mock(
        return_value=httpx.Response(200, json={
            "data": [{
                "id": "giphy-1",
                "title": "lol",
                "slug": "lol-xyz",
                "username": "someuser",
                "user": {"display_name": "Some User"},
                "bitly_url": "https://gph.is/xyz",
                "images": {
                    "original": {
                        "url": "https://media.giphy.com/media/xyz/giphy.gif",
                        "width": "480", "height": "360",
                    },
                },
            }],
        })
    )
    respx.get("https://media.giphy.com/media/xyz/giphy.gif").mock(
        return_value=httpx.Response(200, content=b"GIF89a...", headers={"content-type": "image/gif"})
    )
    results = GiphyProvider(api_key="test-key").search("lol", limit=1)
    assert len(results) == 1
    r = results[0]
    assert r.source_id == "giphy-1"
    assert r.mime_type == "image/gif"
    assert r.data.startswith(b"GIF89a")
    assert r.width == 480
    assert r.height == 360
    assert r.license == "Giphy"
    assert r.attribution == "Some User"


def test_giphy_no_key_raises(monkeypatch):
    monkeypatch.delenv("GIPHY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GIPHY_API_KEY"):
        GiphyProvider(api_key="").search("x")


# ────────── Unsplash ──────────


@respx.mock
def test_unsplash_search():
    respx.get("https://api.unsplash.com/search/photos").mock(
        return_value=httpx.Response(200, json={
            "results": [{
                "id": "unsplash-1",
                "description": "cat on windowsill",
                "alt_description": "cat",
                "width": 4000, "height": 3000,
                "urls": {
                    "regular": "https://images.unsplash.com/photo-1.jpg?w=1080",
                },
                "links": {"html": "https://unsplash.com/photos/abc"},
                "user": {
                    "name": "Jane Photographer",
                    "username": "janephoto",
                    "links": {"html": "https://unsplash.com/@janephoto"},
                },
            }],
        })
    )
    respx.get("https://images.unsplash.com/photo-1.jpg?w=1080").mock(
        return_value=httpx.Response(200, content=b"\xff\xd8\xff jpeg", headers={"content-type": "image/jpeg"})
    )
    results = UnsplashProvider(api_key="test-key").search("cat")
    assert len(results) == 1
    r = results[0]
    assert r.source_id == "unsplash-1"
    assert r.mime_type == "image/jpeg"
    assert r.width == 4000
    assert r.attribution == "Photo by Jane Photographer on Unsplash"
    assert r.license == "Unsplash"


def test_unsplash_no_key_raises(monkeypatch):
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="UNSPLASH_ACCESS_KEY"):
        UnsplashProvider(api_key="").search("x")


@respx.mock
def test_unsplash_sends_auth_header():
    route = respx.get("https://api.unsplash.com/search/photos").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    UnsplashProvider(api_key="my-secret-key").search("x")
    assert route.called
    assert route.calls[0].request.headers["authorization"] == "Client-ID my-secret-key"


# ────────── Pexels ──────────


@respx.mock
def test_pexels_search():
    respx.get("https://api.pexels.com/v1/search").mock(
        return_value=httpx.Response(200, json={
            "photos": [{
                "id": 12345,
                "width": 3000, "height": 2000,
                "url": "https://www.pexels.com/photo/12345/",
                "photographer": "John Snap",
                "photographer_url": "https://www.pexels.com/@johnsnap",
                "src": {
                    "large": "https://images.pexels.com/photos/12345/pexels-photo-12345.jpeg",
                },
                "alt": "Beautiful landscape",
            }],
        })
    )
    respx.get("https://images.pexels.com/photos/12345/pexels-photo-12345.jpeg").mock(
        return_value=httpx.Response(200, content=b"jpeg data", headers={"content-type": "image/jpeg"})
    )
    results = PexelsProvider(api_key="test-key").search("landscape")
    assert len(results) == 1
    r = results[0]
    assert r.source_id == "12345"
    assert r.width == 3000
    assert r.attribution == "Photo by John Snap on Pexels"
    assert r.license == "Pexels"


def test_pexels_no_key_raises(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PEXELS_API_KEY"):
        PexelsProvider(api_key="").search("x")


@respx.mock
def test_pexels_sends_auth_header():
    route = respx.get("https://api.pexels.com/v1/search").mock(
        return_value=httpx.Response(200, json={"photos": []})
    )
    PexelsProvider(api_key="my-key").search("x")
    assert route.calls[0].request.headers["authorization"] == "my-key"


# ────────── Pixabay ──────────


@respx.mock
def test_pixabay_search():
    respx.get("https://pixabay.com/api/").mock(
        return_value=httpx.Response(200, json={
            "hits": [{
                "id": 98765,
                "pageURL": "https://pixabay.com/photos/example-98765/",
                "imageWidth": 1920, "imageHeight": 1280,
                "largeImageURL": "https://pixabay.com/get/large-98765.jpg",
                "webformatURL": "https://pixabay.com/get/web-98765.jpg",
                "user": "JaneArtist",
                "user_id": 42,
                "tags": "cat, cute, animal",
            }],
        })
    )
    respx.get("https://pixabay.com/get/large-98765.jpg").mock(
        return_value=httpx.Response(200, content=b"jpeg bytes", headers={"content-type": "image/jpeg"})
    )
    results = PixabayProvider(api_key="test-key").search("cat")
    assert len(results) == 1
    r = results[0]
    assert r.source_id == "98765"
    assert r.mime_type == "image/jpeg"
    assert r.width == 1920
    assert r.height == 1280
    assert r.attribution == "Image by JaneArtist on Pixabay"
    assert r.license == "Pixabay Content License"
    assert r.metadata["tags"] == "cat, cute, animal"


def test_pixabay_no_key_raises(monkeypatch):
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PIXABAY_API_KEY"):
        PixabayProvider(api_key="").search("x")


@respx.mock
def test_pixabay_sends_key_as_query_param():
    route = respx.get("https://pixabay.com/api/").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    PixabayProvider(api_key="my-pixabay-key").search("x")
    assert route.called
    assert route.calls[0].request.url.params["key"] == "my-pixabay-key"
