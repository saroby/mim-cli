from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mim_cli.ai import GeneratedMetadata
from mim_cli.models import MediaItem
from mim_cli.saver import MetaOverride, save_media, sha256_file
from mim_cli.store import MediaStore


@pytest.fixture
def env(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("MIM_CLI_DIR", str(tmp_store_dir["base"]))
    return tmp_store_dir


@pytest.fixture
def store(env):
    return MediaStore(db_path=env["db"])


@pytest.fixture
def emb_store():
    """ChromaDB는 테스트에서 mock 처리."""
    m = MagicMock()
    return m


@pytest.fixture
def metadata_gen():
    """claude --print subprocess는 mock."""
    m = MagicMock()
    m.generate.return_value = GeneratedMetadata(
        name="AI 이름",
        description="AI 설명",
        tags=["ai태그"],
        emotions=["ai감정"],
        context=["ai맥락"],
    )
    return m


def test_save_bytes_new_item(env, store, emb_store, metadata_gen):
    item, is_new = save_media(
        store=store,
        emb_store=emb_store,
        metadata_gen=metadata_gen,
        data=b"\x89PNG\r\n\x1a\ndata",
        suffix=".png",
        media_type="image",
        source="generated",
        source_provider="leonardo",
        prompt="cute cat",
        model="phoenix",
    )
    assert is_new is True
    assert item.source == "generated"
    assert item.source_provider == "leonardo"
    assert item.prompt == "cute cat"
    assert item.model == "phoenix"
    assert item.content_hash is not None
    assert Path(item.path).exists()
    assert item.name == "AI 이름"
    metadata_gen.generate.assert_called_once()
    emb_store.upsert.assert_called_once()


def test_skip_metadata_no_ai_call(env, store, emb_store, metadata_gen):
    item, _ = save_media(
        store=store,
        emb_store=emb_store,
        metadata_gen=metadata_gen,
        data=b"bytes",
        suffix=".png",
        media_type="image",
        source="generated",
        skip_metadata=True,
    )
    metadata_gen.generate.assert_not_called()
    assert item.description == ""
    assert item.tags == []


def test_full_override_skips_ai(env, store, emb_store, metadata_gen):
    override = MetaOverride(
        name="수동",
        description="수동 설명",
        tags=["t"],
        emotions=["e"],
        context=["c"],
    )
    item, _ = save_media(
        store=store,
        emb_store=emb_store,
        metadata_gen=metadata_gen,
        data=b"bytes",
        suffix=".png",
        media_type="image",
        source="upload",
        meta_override=override,
    )
    metadata_gen.generate.assert_not_called()
    assert item.name == "수동"
    assert item.tags == ["t"]


def test_partial_override_ai_fills_gaps(env, store, emb_store, metadata_gen):
    override = MetaOverride(name="사용자이름")
    item, _ = save_media(
        store=store,
        emb_store=emb_store,
        metadata_gen=metadata_gen,
        data=b"bytes",
        suffix=".png",
        media_type="image",
        source="upload",
        meta_override=override,
    )
    metadata_gen.generate.assert_called_once()
    assert item.name == "사용자이름"         # override 사용
    assert item.description == "AI 설명"      # AI 결과 사용
    assert item.tags == ["ai태그"]


def test_duplicate_by_source_id_returns_existing(env, store, emb_store, metadata_gen):
    first, _ = save_media(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        data=b"a", suffix=".gif", media_type="gif",
        source="fetched", source_provider="giphy", source_id="abc",
    )
    second, is_new = save_media(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        data=b"different bytes",  # 내용은 달라도 source_id 중복
        suffix=".gif", media_type="gif",
        source="fetched", source_provider="giphy", source_id="abc",
    )
    assert is_new is False
    assert second.id == first.id


def test_duplicate_by_hash_returns_existing(env, store, emb_store, metadata_gen):
    data = b"same bytes"
    first, _ = save_media(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        data=data, suffix=".png", media_type="image", source="upload",
    )
    # 다른 source지만 같은 내용 → 해시로 중복 감지
    second, is_new = save_media(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        data=data, suffix=".png", media_type="image", source="upload",
    )
    assert is_new is False
    assert second.id == first.id


def test_force_bypasses_hash_dedup(env, store, emb_store, metadata_gen):
    data = b"same bytes"
    first, _ = save_media(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        data=data, suffix=".png", media_type="image", source="upload",
    )
    second, is_new = save_media(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        data=data, suffix=".png", media_type="image", source="upload",
        force=True,
    )
    assert is_new is True
    assert second.id != first.id
    assert second.content_hash is None  # force=True면 해시 계산 스킵


def test_save_from_existing_path(env, store, emb_store, metadata_gen, tmp_path):
    existing = tmp_path / "input.gif"
    existing.write_bytes(b"gif bytes")
    item, is_new = save_media(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        data=existing, suffix=".gif", media_type="gif", source="upload",
    )
    assert is_new is True
    assert Path(item.path).exists()
    # 원본은 보존
    assert existing.exists()
    # 저장소 내부 경로
    assert str(env["base"]) in item.path


def test_sha256_consistency(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    # sha256("hello") 검증
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
