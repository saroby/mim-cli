import json
import sqlite3
from pathlib import Path
from typing import Callable, Optional

from mim_cli.models import MediaItem


# ────────────────────────────────────────────────────────────
# 스키마 (최신 버전 기준)
# ────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS media_items (
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
);
"""

CREATE_INDEX_SOURCE_KEY = """
CREATE INDEX IF NOT EXISTS idx_source_key
ON media_items(source_provider, source_id);
"""

CREATE_INDEX_CONTENT_HASH = """
CREATE INDEX IF NOT EXISTS idx_content_hash
ON media_items(content_hash);
"""

# 원자적 중복 방지 (부분 유니크 — NULL 값은 중복 허용)
CREATE_UNIQUE_SOURCE_KEY = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_key
ON media_items(source_provider, source_id)
WHERE source_provider IS NOT NULL AND source_id IS NOT NULL;
"""

CREATE_UNIQUE_CONTENT_HASH = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_content_hash
ON media_items(content_hash)
WHERE content_hash IS NOT NULL;
"""

CREATE_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
    id UNINDEXED,
    name,
    description,
    tags,
    emotions,
    context,
    content=media_items,
    content_rowid=rowid,
    tokenize="trigram"
);
"""

SYNC_FTS_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS media_fts_insert
AFTER INSERT ON media_items BEGIN
    INSERT INTO media_fts(rowid, id, name, description, tags, emotions, context)
    VALUES (new.rowid, new.id, new.name, new.description,
            new.tags, new.emotions, new.context);
END;
"""

SYNC_FTS_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS media_fts_delete
AFTER DELETE ON media_items BEGIN
    INSERT INTO media_fts(media_fts, rowid, id, name, description, tags, emotions, context)
    VALUES ('delete', old.rowid, old.id, old.name, old.description,
            old.tags, old.emotions, old.context);
END;
"""

SYNC_FTS_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS media_fts_update
AFTER UPDATE ON media_items BEGIN
    INSERT INTO media_fts(media_fts, rowid, id, name, description, tags, emotions, context)
    VALUES ('delete', old.rowid, old.id, old.name, old.description,
            old.tags, old.emotions, old.context);
    INSERT INTO media_fts(rowid, id, name, description, tags, emotions, context)
    VALUES (new.rowid, new.id, new.name, new.description,
            new.tags, new.emotions, new.context);
END;
"""


# ────────────────────────────────────────────────────────────
# 마이그레이션 (PRAGMA user_version 기반 선형 체인)
# ────────────────────────────────────────────────────────────


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> None:
    """기존 테이블에 출처/계보 컬럼 10개 + 인덱스 2개 추가. 신규 DB는 no-op.

    SQLite는 ALTER TABLE 후 컬럼 존재 여부 확인이 번거로우므로 PRAGMA table_info로 체크.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(media_items)")}
    alters = [
        ("source", "ALTER TABLE media_items ADD COLUMN source TEXT NOT NULL DEFAULT 'upload'"),
        ("source_provider", "ALTER TABLE media_items ADD COLUMN source_provider TEXT"),
        ("source_url", "ALTER TABLE media_items ADD COLUMN source_url TEXT"),
        ("source_id", "ALTER TABLE media_items ADD COLUMN source_id TEXT"),
        ("prompt", "ALTER TABLE media_items ADD COLUMN prompt TEXT"),
        ("model", "ALTER TABLE media_items ADD COLUMN model TEXT"),
        ("content_hash", "ALTER TABLE media_items ADD COLUMN content_hash TEXT"),
        ("attribution", "ALTER TABLE media_items ADD COLUMN attribution TEXT"),
        ("license", "ALTER TABLE media_items ADD COLUMN license TEXT"),
        ("license_url", "ALTER TABLE media_items ADD COLUMN license_url TEXT"),
        ("width", "ALTER TABLE media_items ADD COLUMN width INTEGER"),
        ("height", "ALTER TABLE media_items ADD COLUMN height INTEGER"),
    ]
    for col, sql in alters:
        if col not in existing:
            conn.execute(sql)
    conn.execute(CREATE_INDEX_SOURCE_KEY)
    conn.execute(CREATE_INDEX_CONTENT_HASH)


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """원자적 dedup을 위한 부분 유니크 인덱스. 기존 중복 행이 있으면 마이그레이션 실패.

    실패 시: `mim dedup` 같은 수단으로 수동 정리 후 재시도해야 함.
    """
    conn.execute(CREATE_UNIQUE_SOURCE_KEY)
    conn.execute(CREATE_UNIQUE_CONTENT_HASH)


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    _migrate_v0_to_v1,
    _migrate_v1_to_v2,
]
"""인덱스 = 적용된 후 user_version 값. 즉 MIGRATIONS[0]을 실행하면 user_version이 1이 됨."""


def _apply_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    target = len(MIGRATIONS)
    if current >= target:
        return
    try:
        conn.execute("BEGIN")
        for i in range(current, target):
            MIGRATIONS[i](conn)
            conn.execute(f"PRAGMA user_version = {i + 1}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ────────────────────────────────────────────────────────────
# MediaStore
# ────────────────────────────────────────────────────────────


class MediaStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            # 신규 DB: 최신 스키마로 직접 생성. 기존 DB에는 영향 없음 (IF NOT EXISTS).
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(CREATE_FTS_SQL)
            conn.execute(SYNC_FTS_TRIGGER_INSERT)
            conn.execute(SYNC_FTS_TRIGGER_DELETE)
            conn.execute(SYNC_FTS_TRIGGER_UPDATE)
            # 기존 DB: 누락된 컬럼 ALTER. user_version 갱신.
            _apply_migrations(conn)
            # 신규 DB에서도 user_version을 최신으로 맞춤
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current < len(MIGRATIONS):
                conn.execute(f"PRAGMA user_version = {len(MIGRATIONS)}")
            # 신규 DB를 위해 인덱스도 확실히 생성 (ALTER 경로 안 탔을 때)
            conn.execute(CREATE_INDEX_SOURCE_KEY)
            conn.execute(CREATE_INDEX_CONTENT_HASH)
            conn.execute(CREATE_UNIQUE_SOURCE_KEY)
            conn.execute(CREATE_UNIQUE_CONTENT_HASH)

    def _row_to_item(self, row: sqlite3.Row) -> MediaItem:
        d = dict(row)
        d["tags"] = json.loads(d["tags"])
        d["emotions"] = json.loads(d["emotions"])
        d["context"] = json.loads(d["context"])
        return MediaItem.from_dict(d)

    # ────────── CRUD ──────────

    def save(self, item: MediaItem) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO media_items
                   (id, path, media_type, name, description,
                    tags, emotions, context, created_at, updated_at,
                    source, source_provider, source_url, source_id,
                    prompt, model, content_hash, attribution,
                    license, license_url, width, height)
                   VALUES (?,?,?,?,?,?,?,?,?,?,
                           ?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    item.id, item.path, item.media_type, item.name,
                    item.description,
                    json.dumps(item.tags, ensure_ascii=False),
                    json.dumps(item.emotions, ensure_ascii=False),
                    json.dumps(item.context, ensure_ascii=False),
                    item.created_at, item.updated_at,
                    item.source, item.source_provider, item.source_url,
                    item.source_id, item.prompt, item.model,
                    item.content_hash, item.attribution,
                    item.license, item.license_url,
                    item.width, item.height,
                ),
            )

    _SQLITE_VAR_LIMIT = 999

    def get_many(self, item_ids: list[str]) -> list[MediaItem]:
        if not item_ids:
            return []
        by_id: dict[str, MediaItem] = {}
        with self._conn() as conn:
            for i in range(0, len(item_ids), self._SQLITE_VAR_LIMIT):
                chunk = item_ids[i : i + self._SQLITE_VAR_LIMIT]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT * FROM media_items WHERE id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    by_id[row["id"]] = self._row_to_item(row)
        return [by_id[i] for i in item_ids if i in by_id]

    def get(self, item_id: str) -> Optional[MediaItem]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM media_items WHERE id = ?", (item_id,)
            ).fetchone()
        return self._row_to_item(row) if row else None

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM media_items").fetchone()[0]

    def list_all(
        self,
        media_type: Optional[str] = None,
        source: Optional[str] = None,
        source_provider: Optional[str] = None,
    ) -> list[MediaItem]:
        clauses: list[str] = []
        params: list = []
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if source_provider:
            clauses.append("source_provider = ?")
            params.append(source_provider)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM media_items{where} ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def update(self, item: MediaItem) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE media_items SET
                   name=?, description=?, tags=?, emotions=?, context=?, updated_at=?
                   WHERE id=?""",
                (
                    item.name, item.description,
                    json.dumps(item.tags, ensure_ascii=False),
                    json.dumps(item.emotions, ensure_ascii=False),
                    json.dumps(item.context, ensure_ascii=False),
                    item.updated_at, item.id,
                ),
            )

    def delete(self, item_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM media_items WHERE id = ?", (item_id,))

    # ────────── 중복 판별 ──────────

    def find_by_source(
        self, source_provider: str, source_id: str
    ) -> Optional[MediaItem]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM media_items
                   WHERE source_provider = ? AND source_id = ?
                   LIMIT 1""",
                (source_provider, source_id),
            ).fetchone()
        return self._row_to_item(row) if row else None

    def find_by_hash(self, content_hash: str) -> Optional[MediaItem]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM media_items
                   WHERE content_hash = ?
                   LIMIT 1""",
                (content_hash,),
            ).fetchone()
        return self._row_to_item(row) if row else None
