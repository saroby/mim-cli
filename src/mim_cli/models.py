from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class MediaItem:
    path: str                   # ~/.mim-cli/media/ 내 저장 경로
    media_type: str             # 'video' | 'gif' | 'image'
    name: str                   # 사람이 읽기 쉬운 이름
    description: str            # AI가 생성한 상세 설명
    tags: list[str]             # 검색용 태그
    emotions: list[str]         # 감정 태그 (충격, 기쁨, 슬픔 등)
    context: list[str]          # 사용 맥락 (반응, 전환, 인트로 등)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # 출처/계보 (v1 스키마, 모두 Optional — 기존 데이터 호환)
    source: str = "upload"                        # 'upload' | 'generated' | 'fetched'
    source_provider: Optional[str] = None         # 'leonardo' | 'giphy' | 'openverse' ...
    source_url: Optional[str] = None              # 원본 URL (fetched 전용)
    source_id: Optional[str] = None               # 프로바이더 고유 ID (중복 판별 1차 키)
    prompt: Optional[str] = None                  # 생성 프롬프트 또는 검색 쿼리
    model: Optional[str] = None                   # 생성 모델 별명 (generated 전용)
    content_hash: Optional[str] = None            # 파일 SHA256 (중복 판별 2차 키)
    perceptual_hash: Optional[str] = None         # pHash 기반 시각 중복 판별 키
    attribution: Optional[str] = None             # 저작자 크레딧 문자열
    license: Optional[str] = None                 # 라이선스 식별자 (예: 'CC0', 'Unsplash')
    license_url: Optional[str] = None             # 라이선스 링크
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MediaItem":
        """dict → MediaItem. DB에 알 수 없는 컬럼이 있어도 안전하게 필터링."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def touch(self) -> None:
        """updated_at 갱신"""
        self.updated_at = datetime.now(timezone.utc).isoformat()
