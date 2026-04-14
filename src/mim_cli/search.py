from typing import Optional, TYPE_CHECKING
from mim_cli.models import MediaItem
from mim_cli.store import MediaStore

if TYPE_CHECKING:
    from mim_cli.embeddings import EmbeddingStore


class MediaSearch:
    def __init__(self, store: MediaStore, embedding_store: "Optional[EmbeddingStore]" = None):
        self.store = store
        self.embedding_store = embedding_store

    def query(
        self,
        query: str,
        media_type: Optional[str] = None,
        limit: int = 20,
        source: Optional[str] = None,
        source_provider: Optional[str] = None,
    ) -> list[MediaItem]:
        """FTS5 trigram 검색. 2글자 이하는 LIKE 폴백. source 계열 필터 포함."""
        if len(query.replace(" ", "")) < 3:
            return self._fallback_like_search(
                query, media_type, limit, source, source_provider
            )

        clauses = ["media_fts MATCH ?"]
        params: list = [query]
        if media_type:
            clauses.append("mi.media_type = ?")
            params.append(media_type)
        if source:
            clauses.append("mi.source = ?")
            params.append(source)
        if source_provider:
            clauses.append("mi.source_provider = ?")
            params.append(source_provider)
        where = " AND ".join(clauses)
        sql = (
            f"SELECT mi.* FROM media_fts "
            f"JOIN media_items mi ON media_fts.id = mi.id "
            f"WHERE {where} ORDER BY rank LIMIT ?"
        )
        params.append(limit)
        with self.store._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self.store._row_to_item(r) for r in rows]

    def semantic_query(
        self,
        query: str,
        media_type: Optional[str] = None,
        limit: int = 20,
        source: Optional[str] = None,
        source_provider: Optional[str] = None,
    ) -> list[MediaItem]:
        """벡터 유사도 기반 의미 검색. 필터는 SQLite에서 사후 적용."""
        if not self.embedding_store:
            raise RuntimeError("임베딩 저장소가 초기화되지 않았습니다.")
        ids = self.embedding_store.query(query, media_type=media_type, n_results=limit * 2)
        items = self.store.get_many(ids)

        if source:
            items = [i for i in items if i.source == source]
        if source_provider:
            items = [i for i in items if i.source_provider == source_provider]
        return items[:limit]

    def _fallback_like_search(
        self,
        query: str,
        media_type: Optional[str],
        limit: int,
        source: Optional[str] = None,
        source_provider: Optional[str] = None,
    ) -> list[MediaItem]:
        """짧은 쿼리용 LIKE 폴백."""
        pattern = f"%{query}%"
        clauses = ["(name LIKE ? OR description LIKE ? OR tags LIKE ?)"]
        params: list = [pattern, pattern, pattern]
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if source_provider:
            clauses.append("source_provider = ?")
            params.append(source_provider)
        where = " AND ".join(clauses)
        sql = f"SELECT * FROM media_items WHERE {where} LIMIT ?"
        params.append(limit)
        with self.store._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self.store._row_to_item(r) for r in rows]
