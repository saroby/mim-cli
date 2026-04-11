import pytest
from unittest.mock import MagicMock
from meme_cli.store import MediaStore
from meme_cli.search import MediaSearch
from meme_cli.models import MediaItem
from meme_cli.embeddings import EmbeddingStore


@pytest.fixture
def populated_store(tmp_store_dir):
    store = MediaStore(db_path=tmp_store_dir["db"])
    items = [
        MediaItem(
            path="/tmp/pikachu.png", media_type="image",
            name="놀란 피카츄",
            description="충격받은 피카츄가 입을 벌리고 있는 장면",
            tags=["피카츄", "포켓몬", "충격"],
            emotions=["충격", "놀람"],
            context=["반응", "반전"],
        ),
        MediaItem(
            path="/tmp/explosion.gif", media_type="gif",
            name="폭발 GIF",
            description="건물이 폭발하는 드라마틱한 장면",
            tags=["폭발", "극적", "불"],
            emotions=["극적"],
            context=["전환", "액션"],
        ),
        MediaItem(
            path="/tmp/laugh.mp4", media_type="video",
            name="웃음 클립",
            description="사람들이 크게 웃고 있는 장면",
            tags=["웃음", "기쁨", "재미"],
            emotions=["기쁨", "즐거움"],
            context=["반응", "재미"],
        ),
    ]
    for item in items:
        store.save(item)
    return store


def test_search_by_tag(populated_store):
    search = MediaSearch(populated_store)
    results = search.query("피카츄")
    assert len(results) == 1
    assert results[0].name == "놀란 피카츄"


def test_search_by_emotion(populated_store):
    search = MediaSearch(populated_store)
    results = search.query("충격")
    assert len(results) >= 1
    names = [r.name for r in results]
    assert "놀란 피카츄" in names


def test_search_by_description(populated_store):
    search = MediaSearch(populated_store)
    results = search.query("폭발")
    assert len(results) >= 1
    assert results[0].name == "폭발 GIF"


def test_search_filter_by_type(populated_store):
    search = MediaSearch(populated_store)
    results = search.query("반응", media_type="gif")
    assert all(r.media_type == "gif" for r in results)


def test_search_no_results(populated_store):
    search = MediaSearch(populated_store)
    results = search.query("존재하지않는태그xyz")
    assert results == []


def test_search_short_query_fallback(populated_store):
    """2글자 이하 쿼리는 LIKE 폴백으로 검색"""
    search = MediaSearch(populated_store)
    results = search.query("불")
    assert len(results) >= 1


def test_semantic_query_returns_items(populated_store):
    mock_emb = MagicMock(spec=EmbeddingStore)
    pikachu = [i for i in populated_store.list_all() if i.name == "놀란 피카츄"][0]
    mock_emb.query.return_value = [pikachu.id]

    search = MediaSearch(populated_store, embedding_store=mock_emb)
    results = search.semantic_query("충격받은 표정")

    assert len(results) == 1
    assert results[0].name == "놀란 피카츄"
    mock_emb.query.assert_called_once_with("충격받은 표정", media_type=None, n_results=40)


def test_semantic_query_type_filter(populated_store):
    mock_emb = MagicMock(spec=EmbeddingStore)
    gif = [i for i in populated_store.list_all() if i.media_type == "gif"][0]
    mock_emb.query.return_value = [gif.id]

    search = MediaSearch(populated_store, embedding_store=mock_emb)
    results = search.semantic_query("폭발", media_type="gif")

    mock_emb.query.assert_called_once_with("폭발", media_type="gif", n_results=40)
    assert all(r.media_type == "gif" for r in results)


def test_semantic_query_skips_missing_ids(populated_store):
    """ChromaDB에는 있지만 SQLite에서 삭제된 ID는 무시"""
    mock_emb = MagicMock(spec=EmbeddingStore)
    mock_emb.query.return_value = ["nonexistent-id", populated_store.list_all()[0].id]

    search = MediaSearch(populated_store, embedding_store=mock_emb)
    results = search.semantic_query("테스트")
    assert len(results) == 1


def test_semantic_query_raises_without_embedding_store(populated_store):
    search = MediaSearch(populated_store)
    with pytest.raises(RuntimeError, match="임베딩 저장소"):
        search.semantic_query("테스트")
