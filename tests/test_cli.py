import pytest
import json
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from meme_cli.cli import app
from meme_cli.models import MediaItem
from meme_cli.store import MediaStore

runner = CliRunner()


@pytest.fixture
def env_vars(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("MEME_CLI_DIR", str(tmp_store_dir["base"]))
    return tmp_store_dir


@pytest.fixture
def sample_image(tmp_path):
    img_path = tmp_path / "pikachu.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # 더미 PNG
    return img_path


@pytest.fixture
def store_with_items(env_vars):
    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    items = [
        MediaItem(
            path="/tmp/a.gif", media_type="gif",
            name="폭발", description="폭발 장면", tags=["폭발"],
            emotions=["극적"], context=["액션"],
        ),
        MediaItem(
            path="/tmp/b.png", media_type="image",
            name="충격", description="충격받은 표정", tags=["충격"],
            emotions=["충격"], context=["반응"],
        ),
    ]
    for item in items:
        store.save(item)
    return store


# --- add ---

@patch("meme_cli.saver.MetadataGenerator")
def test_add_command(mock_gen_class, env_vars, sample_image):
    mock_gen = MagicMock()
    mock_gen_class.return_value = mock_gen
    meta = MagicMock()
    meta.name = "노란 이미지"
    meta.description = "노란색 배경"
    meta.tags = ["노란색"]
    meta.emotions = ["중립"]
    meta.context = ["배경"]
    mock_gen.generate.return_value = meta

    # 기본은 JSON 출력
    result = runner.invoke(app, ["add", str(sample_image)])
    assert result.exit_code == 0
    assert "노란 이미지" in result.output


@patch("meme_cli.saver.MetadataGenerator")
def test_add_copies_file_to_store(mock_gen_class, env_vars, sample_image):
    mock_gen = MagicMock()
    mock_gen_class.return_value = mock_gen
    meta = MagicMock()
    meta.name = "테스트"
    meta.description = "설명"
    meta.tags = []
    meta.emotions = []
    meta.context = []
    mock_gen.generate.return_value = meta

    runner.invoke(app, ["add", str(sample_image)])

    media_dir = Path(env_vars["base"]) / "media"
    stored_files = list(media_dir.iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].suffix == ".png"


# --- search ---

def test_search_command_text_output(store_with_items, env_vars):
    result = runner.invoke(app, ["--pretty", "search", "폭발"])
    assert result.exit_code == 0
    assert "폭발" in result.output


def test_search_command_json_output(store_with_items, env_vars):
    # JSON이 기본
    result = runner.invoke(app, ["search", "충격"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["name"] == "충격"
    assert "path" in data[0]


def test_search_no_results(store_with_items, env_vars):
    # JSON 모드에선 빈 배열 "[]"
    result = runner.invoke(app, ["search", "존재안함xyz"])
    assert result.exit_code == 0
    assert json.loads(result.output) == []


# --- edit ---

def test_edit_command_name(store_with_items, env_vars):
    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    items = store.list_all()
    item_id = items[0].id

    result = runner.invoke(app, ["edit", item_id, "--name", "새 이름"])
    assert result.exit_code == 0

    updated = store.get(item_id)
    assert updated.name == "새 이름"


def test_edit_command_tags(store_with_items, env_vars):
    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    items = store.list_all()
    item_id = items[0].id

    result = runner.invoke(app, ["edit", item_id, "--tags", "새태그,다른태그"])
    assert result.exit_code == 0

    updated = store.get(item_id)
    assert "새태그" in updated.tags
    assert "다른태그" in updated.tags


def test_edit_nonexistent(env_vars):
    result = runner.invoke(app, ["edit", "nonexistent-id", "--name", "test"])
    assert result.exit_code != 0


# --- list ---

def test_list_command(store_with_items, env_vars):
    result = runner.invoke(app, ["--pretty", "list"])
    assert result.exit_code == 0
    assert "폭발" in result.output
    assert "충격" in result.output


def test_list_json(store_with_items, env_vars):
    # JSON이 기본
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 2


# --- get ---

def test_get_command(store_with_items, env_vars):
    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    item = store.list_all()[0]

    result = runner.invoke(app, ["--pretty", "get", item.id])
    assert result.exit_code == 0
    assert item.name in result.output


def test_get_json(store_with_items, env_vars):
    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    item = store.list_all()[0]

    # JSON이 기본
    result = runner.invoke(app, ["get", item.id])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["id"] == item.id
    assert "path" in data


# --- remove ---

def test_remove_command(store_with_items, env_vars):
    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    item = store.list_all()[0]

    result = runner.invoke(app, ["--yes", "remove", item.id])
    assert result.exit_code == 0
    assert store.get(item.id) is None


# --- 임베딩 연동 ---

@patch("meme_cli.cli.EmbeddingStore")
@patch("meme_cli.saver.MetadataGenerator")
def test_add_calls_embedding_upsert(mock_gen_class, mock_emb_class, env_vars, sample_image):
    mock_gen = MagicMock()
    mock_gen_class.return_value = mock_gen
    meta = MagicMock()
    meta.name = "테스트"
    meta.description = "설명"
    meta.tags = ["태그"]
    meta.emotions = ["감정"]
    meta.context = ["맥락"]
    mock_gen.generate.return_value = meta

    mock_emb = MagicMock()
    mock_emb_class.return_value = mock_emb

    runner.invoke(app, ["add", str(sample_image)])
    mock_emb.upsert.assert_called_once()


@patch("meme_cli.cli.EmbeddingStore")
def test_edit_calls_embedding_upsert(mock_emb_class, store_with_items, env_vars):
    mock_emb = MagicMock()
    mock_emb_class.return_value = mock_emb

    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    item_id = store.list_all()[0].id

    runner.invoke(app, ["edit", item_id, "--name", "수정된 이름"])
    mock_emb.upsert.assert_called_once()


@patch("meme_cli.cli.EmbeddingStore")
def test_remove_calls_embedding_delete(mock_emb_class, store_with_items, env_vars):
    mock_emb = MagicMock()
    mock_emb_class.return_value = mock_emb

    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    item_id = store.list_all()[0].id

    runner.invoke(app, ["--yes", "remove", item_id])
    mock_emb.delete.assert_called_once_with(item_id)


@patch("meme_cli.cli.EmbeddingStore")
def test_search_semantic_flag(mock_emb_class, store_with_items, env_vars):
    mock_emb = MagicMock()
    mock_emb_class.return_value = mock_emb
    store = MediaStore(db_path=Path(env_vars["base"]) / "memes.db")
    충격_id = next(i.id for i in store.list_all() if i.name == "충격")
    mock_emb.query.return_value = [충격_id]

    result = runner.invoke(app, ["--pretty", "search", "슬픔", "--semantic"])
    assert result.exit_code == 0
    assert "충격" in result.output
