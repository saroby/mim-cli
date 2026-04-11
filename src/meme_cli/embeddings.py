from pathlib import Path
from typing import Optional

import chromadb

from meme_cli.models import MediaItem

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

    def upsert(self, item: MediaItem) -> None:
        text = metadata_to_text(item)
        embedding = self.model.encode(text, normalize_embeddings=True).tolist()
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
        """유사도 높은 순서로 item ID 목록 반환"""
        embedding = self.model.encode(query_text, normalize_embeddings=True).tolist()
        query_params: dict = {
            "query_embeddings": [embedding],
            "n_results": n_results,
        }
        if media_type:
            query_params["where"] = {"media_type": media_type}
        results = self.collection.query(**query_params)
        return results["ids"][0] if results["ids"] else []
