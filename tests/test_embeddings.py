import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from mim_cli.embeddings import EmbeddingStore, metadata_to_text
from mim_cli.models import MediaItem


def make_item(**kwargs) -> MediaItem:
    defaults = dict(
        path="/tmp/test.gif", media_type="gif",
        name="충격 밈", description="충격받은 표정",
        tags=["충격", "반응"], emotions=["충격"], context=["반응"],
    )
    defaults.update(kwargs)
    return MediaItem(**defaults)


def test_metadata_to_text():
    item = make_item()
    text = metadata_to_text(item)
    assert "충격 밈" in text
    assert "충격받은 표정" in text
    assert "충격" in text


@pytest.fixture
def mock_store(tmp_path):
    """chromadb와 SentenceTransformer를 mock한 EmbeddingStore"""
    with patch("mim_cli.embeddings.chromadb") as mock_chroma:
        mock_collection = MagicMock()
        mock_chroma.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection
        store = EmbeddingStore(chroma_dir=tmp_path / "chroma")
        # 모델 lazy load 우회
        store._model = MagicMock()
        store._model.encode.return_value = np.ones(1024, dtype=np.float32)
        yield store, mock_collection


def test_upsert_calls_collection(mock_store):
    store, mock_collection = mock_store
    item = make_item()
    store.upsert(item)
    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args[1]
    assert call_kwargs["ids"] == [item.id]
    assert "embeddings" in call_kwargs
    assert call_kwargs["metadatas"][0]["media_type"] == "gif"


def test_delete_calls_collection(mock_store):
    store, mock_collection = mock_store
    store.delete("some-id")
    mock_collection.delete.assert_called_once_with(ids=["some-id"])


def test_query_returns_ids(mock_store):
    store, mock_collection = mock_store
    mock_collection.query.return_value = {"ids": [["id-1", "id-2"]]}
    result = store.query("충격")
    assert result == ["id-1", "id-2"]


def test_query_with_media_type_filter(mock_store):
    store, mock_collection = mock_store
    mock_collection.query.return_value = {"ids": [["id-1"]]}
    store.query("충격", media_type="gif")
    call_kwargs = mock_collection.query.call_args[1]
    assert call_kwargs["where"] == {"media_type": "gif"}


def test_query_without_filter_has_no_where(mock_store):
    store, mock_collection = mock_store
    mock_collection.query.return_value = {"ids": [[]]}
    store.query("충격")
    call_kwargs = mock_collection.query.call_args[1]
    assert "where" not in call_kwargs
