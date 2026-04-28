import sqlite3
import pytest
from mim_cli.store import MediaStore
from mim_cli.models import MediaItem
from pathlib import Path


@pytest.fixture
def store(tmp_store_dir):
    return MediaStore(db_path=tmp_store_dir["db"])


def make_item(**kwargs) -> MediaItem:
    defaults = dict(
        path="/tmp/test.gif",
        media_type="gif",
        name="테스트",
        description="설명",
        tags=["tag1"],
        emotions=["happy"],
        context=["reaction"],
    )
    defaults.update(kwargs)
    return MediaItem(**defaults)


def test_save_and_get(store):
    item = make_item(name="피카츄")
    store.save(item)
    fetched = store.get(item.id)
    assert fetched is not None
    assert fetched.name == "피카츄"
    assert fetched.tags == ["tag1"]


def test_get_nonexistent_returns_none(store):
    assert store.get("nonexistent-id") is None


def test_list_all(store):
    store.save(make_item(name="A"))
    store.save(make_item(name="B"))
    items = store.list_all()
    assert len(items) == 2


def test_update(store):
    item = make_item(name="원래 이름")
    store.save(item)
    item.name = "바뀐 이름"
    item.touch()
    store.update(item)
    fetched = store.get(item.id)
    assert fetched.name == "바뀐 이름"


def test_delete(store):
    item = make_item()
    store.save(item)
    store.delete(item.id)
    assert store.get(item.id) is None


def test_list_by_type(store):
    store.save(make_item(media_type="gif", name="G"))
    store.save(make_item(media_type="image", name="I"))
    gifs = store.list_all(media_type="gif")
    assert len(gifs) == 1
    assert gifs[0].name == "G"


# ────────── v1 스키마: 출처/계보 ──────────


def test_save_and_get_new_fields(store):
    item = make_item(
        name="생성물",
        source="generated",
        source_provider="leonardo",
        prompt="cute cat",
        model="phoenix",
        content_hash="abc123",
        perceptual_hash="8055005500550055",
        width=1024,
        height=1024,
    )
    store.save(item)
    fetched = store.get(item.id)
    assert fetched.source == "generated"
    assert fetched.source_provider == "leonardo"
    assert fetched.prompt == "cute cat"
    assert fetched.model == "phoenix"
    assert fetched.content_hash == "abc123"
    assert fetched.perceptual_hash == "8055005500550055"
    assert fetched.width == 1024


def test_default_source_is_upload(store):
    item = make_item(name="기본값")
    store.save(item)
    fetched = store.get(item.id)
    assert fetched.source == "upload"
    assert fetched.source_provider is None
    assert fetched.content_hash is None


def test_find_by_source(store):
    store.save(make_item(source="fetched", source_provider="giphy", source_id="abc"))
    store.save(make_item(source="fetched", source_provider="giphy", source_id="xyz"))
    found = store.find_by_source("giphy", "abc")
    assert found is not None
    assert found.source_id == "abc"
    assert store.find_by_source("giphy", "missing") is None


def test_find_by_hash(store):
    store.save(make_item(content_hash="hash-1"))
    found = store.find_by_hash("hash-1")
    assert found is not None
    assert found.content_hash == "hash-1"
    assert store.find_by_hash("hash-404") is None


def test_count_by_prompt_and_find_oldest(store):
    """같은 (provider, prompt) 누적 카운트 + 가장 오래된 항목 조회."""
    store.save(make_item(
        source_provider="pexels", source_id="p-1",
        prompt="cat", created_at="2024-01-01T00:00:00+00:00",
    ))
    store.save(make_item(
        source_provider="pexels", source_id="p-2",
        prompt="cat", created_at="2024-01-02T00:00:00+00:00",
    ))
    store.save(make_item(
        source_provider="pexels", source_id="p-3",
        prompt="dog", created_at="2024-01-03T00:00:00+00:00",
    ))
    store.save(make_item(
        source_provider="giphy", source_id="g-1",
        prompt="cat", created_at="2024-01-04T00:00:00+00:00",
    ))

    assert store.count_by_prompt("pexels", "cat") == 2
    assert store.count_by_prompt("pexels", "dog") == 1
    assert store.count_by_prompt("giphy", "cat") == 1
    assert store.count_by_prompt("giphy", "missing") == 0

    oldest = store.find_oldest_by_prompt("pexels", "cat")
    assert oldest is not None
    assert oldest.source_id == "p-1"
    assert store.find_oldest_by_prompt("pexels", "missing") is None


def test_migration_v2_to_v3_adds_perceptual_hash(tmp_store_dir):
    """v2 DB는 perceptual_hash 컬럼과 인덱스를 자동으로 받아야 한다."""
    db_path = tmp_store_dir["db"]
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE media_items (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            tags TEXT NOT NULL,
            emotions TEXT NOT NULL,
            context TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'upload',
            source_provider TEXT,
            source_url TEXT,
            source_id TEXT,
            prompt TEXT,
            model TEXT,
            content_hash TEXT,
            attribution TEXT,
            license TEXT,
            license_url TEXT,
            width INTEGER,
            height INTEGER
        )
    """)
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    store = MediaStore(db_path=db_path)

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(media_items)")}
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(media_items)")}
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()

    from mim_cli.store import MIGRATIONS
    assert "perceptual_hash" in columns
    assert "idx_perceptual_hash" in indexes
    assert version == len(MIGRATIONS)

    store.save(make_item(name="pHash", perceptual_hash="8055005500550055"))
    fetched = store.list_all()[0]
    assert fetched.perceptual_hash == "8055005500550055"


def test_list_by_source_filter(store):
    store.save(make_item(source="upload", name="u"))
    store.save(make_item(source="generated", source_provider="leonardo", name="g"))
    store.save(make_item(source="fetched", source_provider="giphy", name="f"))
    gens = store.list_all(source="generated")
    assert len(gens) == 1 and gens[0].name == "g"
    giphys = store.list_all(source_provider="giphy")
    assert len(giphys) == 1 and giphys[0].name == "f"


# ────────── 마이그레이션 회귀 ──────────


def test_migration_v0_to_v1(tmp_store_dir):
    """구 v0 스키마 DB가 자동으로 v1으로 업그레이드되고 데이터가 보존되어야 한다."""
    db_path = tmp_store_dir["db"]
    # v0 스키마 DB 수동 생성
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE media_items (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            tags TEXT NOT NULL,
            emotions TEXT NOT NULL,
            context TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute(
        """INSERT INTO media_items
           (id, path, media_type, name, description, tags, emotions, context,
            created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        ("old-id", "/tmp/old.gif", "gif", "구데이터", "설명", "[]", "[]", "[]",
         "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    )
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()

    # MediaStore 초기화 → 마이그레이션 자동 실행
    store = MediaStore(db_path=db_path)

    # 기존 데이터 그대로 읽힘
    item = store.get("old-id")
    assert item is not None
    assert item.name == "구데이터"
    assert item.source == "upload"  # DEFAULT 적용
    assert item.source_provider is None

    # user_version이 최신으로 업데이트되었는지 확인
    conn = sqlite3.connect(db_path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    from mim_cli.store import MIGRATIONS
    assert version == len(MIGRATIONS)

    # 새 필드를 쓰는 아이템도 저장 가능
    store.save(make_item(source="generated", source_provider="leonardo", prompt="test"))
    gens = store.list_all(source="generated")
    assert len(gens) == 1
