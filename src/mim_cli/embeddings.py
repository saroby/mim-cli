from pathlib import Path
from typing import Optional

import chromadb

from mim_cli.models import MediaItem

MODEL_NAME = "BAAI/bge-m3"


def metadata_to_text(item: MediaItem) -> str:
    """MediaItem 메타데이터를 임베딩용 단일 텍스트로 변환"""
    return " ".join([
        item.name,
        item.description,
        " ".join(item.tags),
        " ".join(item.emotions),
        " ".join(item.context),
    ])


class EmbeddingStore:
    def __init__(self, chroma_dir: Path):
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            "memes",
            metadata={"hnsw:space": "cosine"},
        )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def _encode(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def _query_collection(
        self,
        embedding: list[float],
        media_type: Optional[str],
        n_results: int,
    ) -> tuple[list[str], list[float]]:
        params: dict = {"query_embeddings": [embedding], "n_results": n_results}
        if media_type:
            params["where"] = {"media_type": media_type}
        results = self.collection.query(**params)
        ids = (results.get("ids") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        return ids, dists

    def upsert(self, item: MediaItem, *, embedding: Optional[list[float]] = None) -> None:
        """임베딩 upsert. 호출자가 직전에 같은 텍스트를 인코딩했다면 `embedding` 인자로 전달해 재계산 회피."""
        if embedding is None:
            embedding = self._encode(metadata_to_text(item))
        self.collection.upsert(
            ids=[item.id],
            embeddings=[embedding],
            metadatas=[{"media_type": item.media_type}],
        )

    def delete(self, item_id: str) -> None:
        self.collection.delete(ids=[item_id])

    def query(
        self,
        query_text: str,
        media_type: Optional[str] = None,
        n_results: int = 20,
    ) -> list[str]:
        """유사도 높은 순서로 item ID 목록 반환."""
        ids, _ = self._query_collection(
            self._encode(query_text), media_type, n_results
        )
        return ids

    def query_with_distance(
        self,
        query_text: str,
        media_type: Optional[str] = None,
        n_results: int = 5,
        exclude_ids: Optional[list[str]] = None,
    ) -> tuple[list[tuple[str, float]], list[float]]:
        """(matches, query_embedding) 반환. matches는 (item_id, cosine_distance) 페어.

        cosine_distance = 1 − cosine_similarity (0=동일, 1=무관, 2=정반대).
        embedding을 같이 반환해 호출자가 곧 같은 텍스트를 upsert할 때 model.encode 재호출 회피.
        """
        embedding = self._encode(query_text)
        ids, dists = self._query_collection(embedding, media_type, n_results)
        excluded = set(exclude_ids or [])
        matches = [
            (item_id, float(dist))
            for item_id, dist in zip(ids, dists)
            if item_id not in excluded
        ]
        return matches, embedding
