from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GeneratedImage:
    path: Path
    prompt: str
    provider: str
    model: str


@dataclass
class FetchedMedia:
    """온라인 API에서 가져온 미디어 한 건."""

    data: bytes                         # 실제 바이트
    mime_type: str                      # "image/png", "image/gif" 등
    source_url: str                     # 원본 URL (재다운로드 가능)
    source_id: str                      # 프로바이더 고유 ID (중복 판별 키)
    width: Optional[int] = None
    height: Optional[int] = None
    attribution: Optional[str] = None   # 저작자 크레딧
    license: Optional[str] = None       # "CC0", "Unsplash", "Giphy" 등
    license_url: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)  # 프로바이더 고유 정보


class ImageProvider(ABC):
    """이미지 생성 프로바이더 기본 클래스."""

    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        width: int | None = None,
        height: int | None = None,
        aspect_ratio: str | None = None,
    ) -> GeneratedImage:
        ...

    @abstractmethod
    def check_auth(self) -> bool:
        """API 키 등 인증 상태 확인."""
        ...

    def balance(self) -> dict[str, Any]:
        """계정 잔여 크레딧/토큰 조회.

        표준 반환 스키마 (모든 필드 optional, 없으면 None):
          - provider: str  (프로바이더 이름)
          - available: bool  (계정이 활성 상태인지)
          - quota_remaining: int | None  (남은 생성 가능 횟수/토큰)
          - quota_total: int | None  (총 쿼터)
          - unit: str | None  (e.g. "tokens", "credits")
          - renews_at: str | None  (ISO 날짜)
          - plan: str | None
          - note: str | None  (추가 설명)
          - details: dict  (프로바이더 고유 원시 정보)
        """
        raise NotImplementedError(f"{self.name}는 잔여 조회를 지원하지 않습니다.")

    def list_models(self) -> list[dict[str, Any]]:
        """프로바이더가 지원하는 모델 별명 목록.

        각 항목: {"alias": str, "id": str, "note": str | None}
        """
        return []


class FetchProvider(ABC):
    """온라인 무료 미디어 API 베이스. 생성이 아닌 검색/가져오기용."""

    name: str

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 1,
        media_type: Optional[str] = None,  # "gif" | "image" | "video" | None
    ) -> list[FetchedMedia]:
        ...

    @abstractmethod
    def check_auth(self) -> bool:
        """API 키가 설정되어 있는지 (키 불필요한 API는 항상 True)."""
        ...
