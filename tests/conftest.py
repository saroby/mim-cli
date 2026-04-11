import pytest
from pathlib import Path


@pytest.fixture
def tmp_store_dir(tmp_path: Path):
    """격리된 임시 저장소 디렉토리 반환"""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    db_path = tmp_path / "memes.db"
    return {"base": tmp_path, "media": media_dir, "db": db_path}
