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


# ─────────────────────────────────────────────────────────────────────────
# 옵션 2: prompt-saturation 가드 (같은 (provider, prompt) 누적 상한)
# ─────────────────────────────────────────────────────────────────────────


def test_max_per_prompt_returns_existing_after_threshold(env, store, emb_store, metadata_gen):
    """같은 prompt+provider 조합이 max_per_prompt 도달 시 신규 저장 거부."""
    common = dict(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels",
        prompt="휴대폰 보는 사람",
        max_per_prompt=2,
    )
    first, _ = save_media(data=b"image1", source_id="p-1", **common)
    second, _ = save_media(data=b"image2", source_id="p-2", **common)
    third, third_is_new = save_media(data=b"image3", source_id="p-3", **common)

    # 3번째는 거부되고 첫 번째(가장 오래된) 항목이 반환됨
    assert third_is_new is False
    assert third.id == first.id
    # DB에는 2건만 저장됨
    assert len([i for i in store.list_all() if i.prompt == "휴대폰 보는 사람"]) == 2


def test_max_per_prompt_disabled_when_zero(env, store, emb_store, metadata_gen):
    """max_per_prompt=0/None이면 가드 비활성, 무제한 누적."""
    common = dict(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels",
        prompt="질문",
        max_per_prompt=0,
    )
    save_media(data=b"a", source_id="p-1", **common)
    save_media(data=b"b", source_id="p-2", **common)
    _, third_is_new = save_media(data=b"c", source_id="p-3", **common)
    assert third_is_new is True


def test_max_per_prompt_force_bypass(env, store, emb_store, metadata_gen):
    """force=True면 saturation 가드 우회."""
    common = dict(
        store=store, emb_store=emb_store, metadata_gen=metadata_gen,
        suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels",
        prompt="동일 prompt",
        max_per_prompt=1,
    )
    save_media(data=b"a", source_id="p-1", **common)
    _, second_is_new = save_media(
        data=b"b", source_id="p-2", **common, force=True,
    )
    assert second_is_new is True


# ─────────────────────────────────────────────────────────────────────────
# 옵션 1: semantic dedup (AI 메타 임베딩 cosine distance)
# ─────────────────────────────────────────────────────────────────────────


def test_semantic_dedup_returns_existing_when_close(env, store, metadata_gen):
    """기존 임베딩과 cosine distance ≤ threshold면 신규 저장 거부."""
    emb = MagicMock()
    # 첫 호출은 빈 결과 (DB 비어있음), 두 번째 호출에서 가까운 매치를 반환하도록 side_effect 설정
    pending_matches: list[tuple[str, float]] = []
    # query_with_distance는 (matches, embedding) 튜플 반환
    emb.query_with_distance.side_effect = lambda *a, **kw: (list(pending_matches), [0.0])

    first, first_is_new = save_media(
        store=store, emb_store=emb, metadata_gen=metadata_gen,
        data=b"first", suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels", source_id="p-1",
        prompt="cat",
        semantic_threshold=0.05,
    )
    assert first_is_new is True

    # 두 번째 시도에서 첫 번째 항목과 매우 가까운 거리(0.01)를 반환
    pending_matches.append((first.id, 0.01))

    second, second_is_new = save_media(
        store=store, emb_store=emb, metadata_gen=metadata_gen,
        data=b"different bytes but semantic dup",
        suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels", source_id="p-2",
        prompt="kitten",
        semantic_threshold=0.05,
    )
    assert second_is_new is False
    assert second.id == first.id


def test_semantic_dedup_passes_when_far(env, store, metadata_gen):
    """cosine distance > threshold면 신규 저장."""
    emb = MagicMock()
    pending_matches: list[tuple[str, float]] = []
    # query_with_distance는 (matches, embedding) 튜플 반환
    emb.query_with_distance.side_effect = lambda *a, **kw: (list(pending_matches), [0.0])

    first, _ = save_media(
        store=store, emb_store=emb, metadata_gen=metadata_gen,
        data=b"first", suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels", source_id="p-1",
        prompt="cat",
        semantic_threshold=0.05,
    )

    # 거리 0.5 — threshold 0.05 초과 → 통과
    pending_matches.append((first.id, 0.5))

    second, second_is_new = save_media(
        store=store, emb_store=emb, metadata_gen=metadata_gen,
        data=b"second", suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels", source_id="p-2",
        prompt="dog",
        semantic_threshold=0.05,
    )
    assert second_is_new is True
    assert second.id != first.id


def test_semantic_dedup_force_bypass(env, store, metadata_gen):
    """force=True면 semantic dedup도 우회."""
    emb = MagicMock()
    pending_matches: list[tuple[str, float]] = []
    # query_with_distance는 (matches, embedding) 튜플 반환
    emb.query_with_distance.side_effect = lambda *a, **kw: (list(pending_matches), [0.0])

    first, _ = save_media(
        store=store, emb_store=emb, metadata_gen=metadata_gen,
        data=b"a", suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels", source_id="p-1",
        prompt="cat",
    )
    pending_matches.append((first.id, 0.0))

    _, second_is_new = save_media(
        store=store, emb_store=emb, metadata_gen=metadata_gen,
        data=b"b", suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels", source_id="p-2",
        prompt="cat",
        force=True,
    )
    assert second_is_new is True


def test_semantic_dedup_skipped_when_metadata_empty(env, store, metadata_gen):
    """skip_metadata=True 또는 메타가 비어있으면 semantic 비교 자체를 안 함."""
    emb = MagicMock()
    save_media(
        store=store, emb_store=emb, metadata_gen=metadata_gen,
        data=b"a", suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels", source_id="p-1",
        prompt="cat",
        skip_metadata=True,
        semantic_threshold=0.05,
    )
    emb.query_with_distance.assert_not_called()


def test_semantic_dedup_disabled_when_threshold_none(env, store, metadata_gen):
    """semantic_threshold=None이면 임베딩 조회 자체를 스킵."""
    emb = MagicMock()
    save_media(
        store=store, emb_store=emb, metadata_gen=metadata_gen,
        data=b"a", suffix=".jpg", media_type="image",
        source="fetched", source_provider="pexels", source_id="p-1",
        prompt="cat",
        semantic_threshold=None,
    )
    emb.query_with_distance.assert_not_called()
