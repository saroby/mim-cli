"""통합 저장 헬퍼. add/generate/fetch 세 명령이 공유.

흐름:
  1. (source_provider, source_id)로 1차 중복 체크
  2. tempfile에 바이트 쓰기 or 파일 복사
  3. SHA256 해시 계산 (force=True면 스킵)
  4. content_hash로 2차 중복 체크 (중복이면 temp 삭제 후 기존 아이템 반환)
  5. 최종 경로 (media_dir/{uuid}.{ext})로 이동
  6. meta_override 기준으로 AI 메타데이터(claude --print) 호출 여부 판단
     - 전부 채워져 있으면 AI 스킵
     - 부분 채움이면 claude --print로 분석 후 비어있는 필드만 채움
     - skip_metadata=True면 무조건 AI 스킵 (빈 값 허용)
  7. MediaItem 조립 + store.save + emb_store.upsert
  8. (MediaItem, is_new) 반환

주의:
  - AI 호출(이미지 분석)은 오직 `claude --print` subprocess (ai.MetadataGenerator)만 사용.
  - 이미지 생성은 별개의 프로바이더 API (gemini/leonardo/replicate)를 사용.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from meme_cli.ai import GeneratedMetadata, MetadataGenerator
from meme_cli.config import get_media_dir
from meme_cli.embeddings import EmbeddingStore
from meme_cli.models import MediaItem
from meme_cli.store import MediaStore


@dataclass
class MetaOverride:
    """사용자가 CLI 옵션으로 전달한 메타데이터. None 필드는 AI가 채움."""

    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    emotions: Optional[list[str]] = None
    context: Optional[list[str]] = None

    def is_complete(self) -> bool:
        """모든 필드가 채워졌는지. 전부 채워지면 AI 호출 스킵 가능."""
        return all(
            v is not None
            for v in (
                self.name, self.description, self.tags,
                self.emotions, self.context,
            )
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# 확장자 검증: `.` + 영숫자 1~8자. 외부 파일명에서 넘어오는 suffix가
# LLM 프롬프트/파일명에 그대로 꽂히는 걸 막음.
_SAFE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


def _sanitize_suffix(suffix: str, fallback: str = ".bin") -> str:
    if suffix and _SAFE_SUFFIX_RE.match(suffix):
        return suffix.lower()
    return fallback


def save_media(
    *,
    store: MediaStore,
    emb_store: EmbeddingStore,
    metadata_gen: Optional[MetadataGenerator] = None,
    data: bytes | Path,
    suffix: str,
    media_type: str,
    source: str,
    source_provider: Optional[str] = None,
    source_url: Optional[str] = None,
    source_id: Optional[str] = None,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    attribution: Optional[str] = None,
    license: Optional[str] = None,
    license_url: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    meta_override: Optional[MetaOverride] = None,
    skip_metadata: bool = False,
    force: bool = False,
) -> tuple[MediaItem, bool]:
    """미디어를 저장소에 저장. (item, is_new) 반환.

    중복이면 기존 item 반환 + is_new=False. 신규면 새 item + is_new=True.

    Args:
        data: 바이트(신규 다운로드) 또는 이미 존재하는 파일 경로(업로드).
        suffix: 확장자 ".png", ".gif" 등. data가 Path면 path.suffix 사용 가능.
        source: "upload" | "generated" | "fetched"
        skip_metadata: True면 AI 메타데이터 호출 완전 스킵 (빈 값으로 저장).
        force: True면 content_hash 중복 무시하고 강제 저장 (add --force).
    """
    meta_override = meta_override or MetaOverride()
    suffix = _sanitize_suffix(suffix)

    if source_provider and source_id:
        existing = store.find_by_source(source_provider, source_id)
        if existing:
            return existing, False

    media_dir = get_media_dir()
    media_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, Path):
        src_path = data
        temp_path: Optional[Path] = None
    else:
        tmp = tempfile.NamedTemporaryFile(
            dir=media_dir, suffix=suffix, delete=False
        )
        tmp.write(data)
        tmp.close()
        temp_path = Path(tmp.name)
        src_path = temp_path

    try:
        if force:
            content_hash: Optional[str] = None
        else:
            content_hash = sha256_file(src_path)
            existing = store.find_by_hash(content_hash)
            if existing:
                return existing, False

        dest = media_dir / f"{uuid.uuid4()}{suffix}"
        if temp_path is not None:
            shutil.move(str(temp_path), str(dest))
            temp_path = None
        else:
            shutil.copy2(src_path, dest)

        saved_to_store = False
        try:
            if skip_metadata or meta_override.is_complete():
                meta = GeneratedMetadata(
                    name=meta_override.name or dest.stem[:16],
                    description=meta_override.description or "",
                    tags=meta_override.tags or [],
                    emotions=meta_override.emotions or [],
                    context=meta_override.context or [],
                )
            else:
                gen = metadata_gen or MetadataGenerator()
                ai_meta = gen.generate(dest)
                meta = GeneratedMetadata(
                    name=meta_override.name or ai_meta.name,
                    description=meta_override.description or ai_meta.description,
                    tags=meta_override.tags or ai_meta.tags,
                    emotions=meta_override.emotions or ai_meta.emotions,
                    context=meta_override.context or ai_meta.context,
                )

            item = MediaItem(
                path=str(dest),
                media_type=media_type,
                name=meta.name,
                description=meta.description,
                tags=meta.tags,
                emotions=meta.emotions,
                context=meta.context,
                source=source,
                source_provider=source_provider,
                source_url=source_url,
                source_id=source_id,
                prompt=prompt,
                model=model,
                content_hash=content_hash,
                attribution=attribution,
                license=license,
                license_url=license_url,
                width=width,
                height=height,
            )
            try:
                store.save(item)
                saved_to_store = True
            except sqlite3.IntegrityError:
                # 동시 실행으로 인한 race: 승자 행을 찾아 반환, 내 복사본은 삭제
                winner = None
                if source_provider and source_id:
                    winner = store.find_by_source(source_provider, source_id)
                if winner is None and content_hash:
                    winner = store.find_by_hash(content_hash)
                dest.unlink(missing_ok=True)
                if winner is not None:
                    return winner, False
                raise
            # 임베딩 실패는 치명적이지 않음 — 롤백하면 해시/source_id 중복 검사가 재시도를 막음.
            # 로그만 남기고 DB는 유지. 이후 복구 루틴으로 재임베딩.
            try:
                emb_store.upsert(item)
            except Exception as emb_err:
                import sys as _sys
                from meme_cli.output import mask_secret
                masked = mask_secret(str(emb_err))
                print(f"[saver] 임베딩 업서트 실패 (DB는 유지): {masked}", file=_sys.stderr)
            return item, True
        except Exception:
            if saved_to_store:
                try:
                    store.delete(item.id)
                except Exception:
                    pass
            dest.unlink(missing_ok=True)
            raise

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
