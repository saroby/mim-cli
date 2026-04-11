"""Reddit fetch 프로바이더 mock 테스트."""

import httpx
import respx

from meme_cli.providers.fetch.reddit import RedditProvider


def _search_response(children):
    return {"data": {"children": [{"kind": "t3", "data": d} for d in children]}}


@respx.mock
def test_search_direct_image_url():
    respx.get("https://www.reddit.com/search.json").mock(
        return_value=httpx.Response(200, json=_search_response([{
            "id": "abc1",
            "title": "cute kitten",
            "subreddit": "aww",
            "author": "user1",
            "permalink": "/r/aww/comments/abc1/cute_kitten/",
            "url": "https://i.redd.it/xyz.jpg",
            "score": 500,
            "preview": {"images": [{"source": {"width": 1024, "height": 768}}]},
        }]))
    )
    respx.get("https://i.redd.it/xyz.jpg").mock(
        return_value=httpx.Response(200, content=b"\xff\xd8\xff jpeg", headers={"content-type": "image/jpeg"})
    )
    results = RedditProvider().search("kitten", limit=1)
    assert len(results) == 1
    r = results[0]
    assert r.source_id == "abc1"
    assert r.mime_type == "image/jpeg"
    assert r.width == 1024
    assert r.attribution == "u/user1"
    assert r.license == "Reddit"
    assert r.metadata["subreddit"] == "aww"
    assert r.metadata["score"] == 500


@respx.mock
def test_search_gif_preview_variant():
    respx.get("https://www.reddit.com/search.json").mock(
        return_value=httpx.Response(200, json=_search_response([{
            "id": "gif1",
            "title": "dancing cat",
            "subreddit": "reactiongifs",
            "author": "gifmaker",
            "permalink": "/r/reactiongifs/comments/gif1/",
            "url": "https://v.redd.it/xyz",
            "preview": {
                "images": [{
                    "source": {"url": "https://preview.redd.it/src.jpg", "width": 500, "height": 400},
                    "variants": {
                        "gif": {
                            "source": {
                                "url": "https://preview.redd.it/xyz.gif?format=mp4&amp;s=abc",
                                "width": 500, "height": 400,
                            },
                        },
                    },
                }],
            },
        }]))
    )
    respx.get(
        "https://preview.redd.it/xyz.gif",
        params={"format": "mp4", "s": "abc"},
    ).mock(return_value=httpx.Response(200, content=b"GIF89a", headers={"content-type": "image/gif"}))
    results = RedditProvider().search("dancing", limit=1)
    assert len(results) == 1
    r = results[0]
    assert r.mime_type == "image/gif"
    assert r.width == 500


@respx.mock
def test_search_filters_text_only_posts():
    # 1개는 미디어 있음, 1개는 text-only
    respx.get("https://www.reddit.com/search.json").mock(
        return_value=httpx.Response(200, json=_search_response([
            {
                "id": "text1",
                "title": "discussion post",
                "url": "https://www.reddit.com/r/memes/comments/text1/discussion/",
                "subreddit": "memes",
                # preview 없음, 직접 url도 미디어 아님
            },
            {
                "id": "img1",
                "title": "meme image",
                "url": "https://i.redd.it/meme.png",
                "subreddit": "memes",
            },
        ]))
    )
    respx.get("https://i.redd.it/meme.png").mock(
        return_value=httpx.Response(200, content=b"PNG data", headers={"content-type": "image/png"})
    )
    results = RedditProvider().search("stuff", limit=5)
    assert len(results) == 1
    assert results[0].source_id == "img1"


@respx.mock
def test_search_with_subreddit():
    route = respx.get("https://www.reddit.com/r/dankmemes/search.json").mock(
        return_value=httpx.Response(200, json=_search_response([]))
    )
    RedditProvider(subreddit="dankmemes").search("x")
    assert route.called
    # restrict_sr=on 파라미터 전송 확인
    assert "restrict_sr=on" in str(route.calls[0].request.url)


@respx.mock
def test_user_agent_header_sent():
    route = respx.get("https://www.reddit.com/search.json").mock(
        return_value=httpx.Response(200, json=_search_response([]))
    )
    RedditProvider().search("x")
    ua = route.calls[0].request.headers["user-agent"]
    assert "meme-cli" in ua


@respx.mock
def test_oauth_mode_when_credentials_set(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "app-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "app-secret")

    token_route = respx.post("https://www.reddit.com/api/v1/access_token").mock(
        return_value=httpx.Response(200, json={"access_token": "bearer-tok", "expires_in": 3600})
    )
    search_route = respx.get("https://oauth.reddit.com/search").mock(
        return_value=httpx.Response(200, json=_search_response([]))
    )

    RedditProvider().search("x")

    assert token_route.called
    assert search_route.called
    assert search_route.calls[0].request.headers["authorization"] == "Bearer bearer-tok"


def test_check_auth_always_true():
    assert RedditProvider().check_auth() is True


@respx.mock
def test_media_type_filter_gif():
    respx.get("https://www.reddit.com/search.json").mock(
        return_value=httpx.Response(200, json=_search_response([
            {"id": "img", "url": "https://i.redd.it/a.jpg"},
            {"id": "gif", "url": "https://i.redd.it/b.gif"},
        ]))
    )
    respx.get("https://i.redd.it/b.gif").mock(
        return_value=httpx.Response(200, content=b"GIF89a", headers={"content-type": "image/gif"})
    )
    results = RedditProvider().search("x", limit=5, media_type="gif")
    assert len(results) == 1
    assert results[0].source_id == "gif"
