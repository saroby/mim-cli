from pathlib import Path
import os


def get_base_dir() -> Path:
    """환경변수 MEME_CLI_DIR 또는 ~/.meme-cli 사용"""
    base = os.environ.get("MEME_CLI_DIR")
    if base:
        return Path(base)
    return Path.home() / ".meme-cli"


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
