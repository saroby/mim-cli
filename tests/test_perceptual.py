from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

HAS_IMAGE_DEPS = (
    importlib.util.find_spec("imagehash") is not None
    and importlib.util.find_spec("PIL") is not None
)
pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_DEPS,
    reason="imagehash/Pillow 미설치",
)

if HAS_IMAGE_DEPS:
    from PIL import Image, ImageDraw, ImageEnhance, PngImagePlugin

from mim_cli.ai import GeneratedMetadata
from mim_cli.saver import save_media
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
    return MagicMock()


def _mock_metadata(mock_gen_class):
    mock_gen = MagicMock()
    mock_gen.generate.return_value = GeneratedMetadata(
        name="시각 테스트",
        description="테스트 이미지",
        tags=["테스트"],
        emotions=["중립"],
        context=["회귀"],
    )
    mock_gen_class.return_value = mock_gen
    return mock_gen


def _stock_like_image() -> Image.Image:
    image = Image.new("RGB", (128, 128), (235, 238, 230))
    draw = ImageDraw.Draw(image)
    for x in range(0, 128, 8):
        draw.line((x, 0, 127 - x, 127), fill=(180, 190, 200), width=1)
    draw.rectangle((18, 24, 74, 92), fill=(210, 55, 45))
    draw.ellipse((62, 28, 112, 102), fill=(40, 100, 210))
    draw.line((0, 112, 127, 88), fill=(20, 20, 20), width=4)
    return image


def _different_image() -> Image.Image:
    image = Image.new("RGB", (128, 128), (30, 40, 55))
    draw = ImageDraw.Draw(image)
    for y in range(0, 128, 12):
        color = (240, 210, 40) if (y // 12) % 2 == 0 else (35, 180, 90)
        draw.rectangle((0, y, 127, y + 6), fill=color)
    draw.polygon([(12, 118), (64, 10), (116, 118)], fill=(245, 245, 245))
    draw.rectangle((44, 46, 84, 86), fill=(15, 15, 15))
    return image


def _png_bytes(image: Image.Image, marker: str) -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("marker", marker)
    buffer = BytesIO()
    image.save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def _save_test_media(store, emb_store, data: bytes, source_id: str, *, force: bool = False):
    return save_media(
        store=store,
        emb_store=emb_store,
        data=data,
        suffix=".png",
        media_type="image",
        source="fetched",
        source_provider="pexels",
        source_id=source_id,
        force=force,
    )


@patch("mim_cli.saver.MetadataGenerator")
def test_perceptual_dedup_catches_identical_images(mock_gen_class, env, store, emb_store):
    mock_gen = _mock_metadata(mock_gen_class)
    image = _stock_like_image()

    first, first_is_new = _save_test_media(
        store, emb_store, _png_bytes(image, "first"), "photo-1"
    )
    second, second_is_new = _save_test_media(
        store, emb_store, _png_bytes(image, "second"), "photo-2"
    )

    assert first_is_new is True
    assert second_is_new is False
    assert second.id == first.id
    assert first.perceptual_hash is not None
    assert mock_gen.generate.call_count == 1
    assert len(list(Path(env["media"]).iterdir())) == 1


@patch("mim_cli.saver.MetadataGenerator")
def test_perceptual_dedup_catches_near_duplicate_images(mock_gen_class, store, emb_store):
    _mock_metadata(mock_gen_class)
    image = _stock_like_image()
    near_duplicate = ImageEnhance.Brightness(
        image.resize((120, 120)).resize((128, 128))
    ).enhance(1.04)

    first, _ = _save_test_media(store, emb_store, _png_bytes(image, "base"), "photo-1")
    second, second_is_new = _save_test_media(
        store, emb_store, _png_bytes(near_duplicate, "variant"), "photo-2"
    )

    assert second_is_new is False
    assert second.id == first.id


@patch("mim_cli.saver.MetadataGenerator")
def test_perceptual_dedup_allows_clearly_different_images(mock_gen_class, store, emb_store):
    _mock_metadata(mock_gen_class)

    first, first_is_new = _save_test_media(
        store, emb_store, _png_bytes(_stock_like_image(), "stock"), "photo-1"
    )
    second, second_is_new = _save_test_media(
        store, emb_store, _png_bytes(_different_image(), "different"), "photo-2"
    )

    assert first_is_new is True
    assert second_is_new is True
    assert second.id != first.id


@patch("mim_cli.saver.MetadataGenerator")
def test_force_bypasses_perceptual_dedup(mock_gen_class, store, emb_store):
    _mock_metadata(mock_gen_class)
    image = _stock_like_image()

    first, _ = _save_test_media(store, emb_store, _png_bytes(image, "first"), "photo-1")
    second, second_is_new = _save_test_media(
        store,
        emb_store,
        _png_bytes(image, "second"),
        "photo-2",
        force=True,
    )

    assert second_is_new is True
    assert second.id != first.id
    assert second.content_hash is None
    assert second.perceptual_hash is None
