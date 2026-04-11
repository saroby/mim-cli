# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
uv sync                    # 의존성 설치
uv run meme --help         # CLI 실행
pytest tests/              # 전체 테스트
pytest tests/test_store.py # 단일 파일 테스트
pytest tests/test_store.py::test_save_and_get -v  # 단일 테스트
```

## Architecture

밈 미디어 관리 CLI. Typer + Rich 기반. **AI-first 설계** — 기본 출력이 **JSON** (stdout), 진행/에러는 stderr. 사람이 직접 쓸 때는 `--pretty` 플래그로 Rich UI 활성화.

**AI 사용 규칙:**
- **이미지 생성**: 전용 프로바이더 API (Gemini / Leonardo / Replicate)
- **그 외 모든 AI 작업 (이미지 분석, 메타데이터 추출 등)**: `claude --print` subprocess만 사용 (`ai.py:MetadataGenerator`)

**3가지 미디어 입수 경로 → 단일 저장 흐름:**
1. **`add <file>`** (수동 업로드) — 로컬 파일 복사 → 해시 중복 체크 → `saver.save_media(source="upload")`
2. **`generate <prompt>`** (AI 생성) — 프로바이더 API 호출 → `saver.save_media(source="generated", prompt=..., model=...)`
3. **`fetch <query> -p <provider>`** (온라인 무료 API) — 검색 → 다운로드 → 중복 체크 → `saver.save_media(source="fetched", source_url=..., source_id=..., prompt=<query>, attribution=..., license=...)`

**중복 감지 (자동):**
- 1차: `(source_provider, source_id)` 인덱스 조회 — 온라인 API 재호출 시 빠른 스킵
- 2차: `content_hash` (SHA256) 조회 — 같은 바이트면 provider가 달라도 스킵
- 중복 시 기존 아이템 반환 + `{"skipped": true, "existing_id": "..."}`
- `add --force`로 해시 중복 우회 가능

**데이터 흐름:**
- **Search**: FTS5 trigram 매칭 (3자 미만은 LIKE 폴백) 또는 `--semantic` 플래그로 벡터 유사도 검색. `--source upload|generated|fetched`, `--from-provider <name>` 필터
- **Generate**: 프롬프트 → Provider API → PNG 저장 → (옵션) `saver.save_media`. `--no-save`로 DB 스킵, `--skip-metadata`로 AI 분석 스킵
- **Fetch**: 쿼리 → 프로바이더 search API → 이미지 바이트 다운로드 → `saver.save_media` (중복 스킵)

**핵심 모듈:**
- `store.py` — SQLite CRUD + FTS5 전문검색. `PRAGMA user_version` 기반 선형 마이그레이션 프레임워크 (`MIGRATIONS` 리스트). `find_by_source`, `find_by_hash`로 중복 판별. `get_many()`는 999개 변수 제한 우회를 위해 청크 처리
- `embeddings.py` — ChromaDB + BAAI/bge-m3 (lazy load). 테스트에서는 mock 처리
- `ai.py` — **유일한 non-생성 AI 경로**. `claude --print` subprocess로 이미지 분석 후 JSON 메타데이터 파싱
- `output.py` — 글로벌 `--pretty`/`--timeout`/`--yes` 상태 관리, 시크릿 마스킹 (`AIzaSy*`, `r8_*`, UUID, Bearer), 구조화 에러 분류 (`rate_limited`, `payment_required`, `unauthorized`, `not_found`, `timeout`, `request_failed`), stdout/stderr 분리
- `saver.py` — **3가지 저장 진입점의 통합 흐름**. `save_media(data, suffix, media_type, source, ...)` → `(MediaItem, is_new)`. tempfile → 해시 → 중복 체크 → AI 메타(claude --print, 부분 override 지원) → store.save + embedding upsert
- `providers/` — 두 종류 ABC:
  - `ImageProvider` (생성): Gemini/Leonardo/Replicate. `generate()`, `balance()`, `list_models()`
  - `FetchProvider` (온라인 API): `providers/fetch/` 서브패키지. `search(query, limit, media_type) → list[FetchedMedia]`
- `cli.py` — 두 레지스트리 분리: `GEN_PROVIDERS`, `FETCH_PROVIDERS`. `_build_gen_provider` / `_build_fetch_provider` 분기

**저장소 구조** (`~/.meme-cli/`, `MEME_CLI_DIR` 환경변수로 오버라이드):
- `memes.db` — SQLite (media_items + FTS5 가상 테이블). `PRAGMA user_version` 현재 버전 1
- `media/` — 원본/가져온/생성된 미디어 파일 (UUID 파일명, 확장자는 mime에서 추론)
- `chroma/` — ChromaDB 벡터 저장소

**MediaItem 스키마 (v1):**
- 기본: `id`, `path`, `media_type`, `name`, `description`, `tags`, `emotions`, `context`, `created_at`, `updated_at`
- 출처/계보: `source` (upload|generated|fetched), `source_provider`, `source_url`, `source_id`, `prompt`, `model`, `content_hash`, `attribution`, `license`, `license_url`, `width`, `height`
- `from_dict()`는 **unknown-field 필터** — DB에 향후 새 컬럼 생겨도 역직렬화 안 깨짐

## 글로벌 옵션 (서브커맨드 앞에 배치)

- `--pretty` — 사람용 Rich 출력 (환경변수 `MEME_CLI_PRETTY`). 미지정 시 JSON (AI 친화 기본)
- `--timeout N` — HTTP 타임아웃 초 (환경변수 `MEME_CLI_TIMEOUT`)
- `--yes`/`-y` — 확인 프롬프트 자동 승인 (환경변수 `MEME_CLI_ASSUME_YES`)

## 환경변수

- `MEME_CLI_DIR` — 저장소 루트 경로 (기본: `~/.meme-cli`)
- **생성 프로바이더:**
  - `GEMINI_API_KEYS` — Gemini (쉼표로 여러 키 구분, 429 시 자동 전환)
  - `REPLICATE_API_TOKEN` (또는 `REPLICATE_API_KEY`) — Replicate
  - `LEONARDO_API_KEY` — Leonardo.ai
- **가져오기 프로바이더:**
  - `GIPHY_API_KEY` — https://developers.giphy.com/dashboard/
  - `UNSPLASH_ACCESS_KEY` — https://unsplash.com/developers
  - `PEXELS_API_KEY` — https://www.pexels.com/api/new/
  - `PIXABAY_API_KEY` — https://pixabay.com/api/docs/
  - Openverse는 키 불필요
  - Reddit은 기본 무인증 (분당 10회). 선택: `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET` (OAuth, 분당 100회), `REDDIT_USER_AGENT` (권장), `REDDIT_DEFAULT_SUBREDDIT`

## 테스트 패턴

- `conftest.py`의 `tmp_store_dir` fixture로 테스트별 격리된 DB/미디어 디렉토리 생성
- EmbeddingStore는 모델 로딩이 느리므로 테스트에서 mock 처리
- **Fetch 프로바이더**: `respx` + `httpx.MockTransport`로 네트워크 mock (실제 API 호출 없음)
- **MetadataGenerator** (claude --print): 테스트에서 `@patch("meme_cli.saver.MetadataGenerator")` 로 mock
- 한국어 메타데이터 (이름, 태그, 감정, 맥락)
