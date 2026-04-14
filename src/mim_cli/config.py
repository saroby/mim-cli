from pathlib import Path
import os


def get_base_dir() -> Path:
    """환경변수 MIM_CLI_DIR 또는 ~/.mim-cli 사용"""
    base = os.environ.get("MIM_CLI_DIR")
    if base:
        return Path(base)
    return Path.home() / ".mim-cli"


def get_db_path() -> Path:
    d = get_base_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "memes.db"


def get_media_dir() -> Path:
    d = get_base_dir() / "media"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_chroma_dir() -> Path:
    d = get_base_dir() / "chroma"
    d.mkdir(parents=True, exist_ok=True)
    return d
