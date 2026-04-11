# meme-cli

> **AI 에이전트를 위한 밈 미디어 관리 CLI.**
> 기본 출력이 **JSON (stdout)**, 진행/에러는 stderr로 분리 — AI가 파싱하기 쉬운 형태로 설계됨. 사람이 직접 쓸 때는 `--pretty` 플래그로 Rich UI.

MCP 서버를 띄울 필요 없이, Claude Code·Cursor·Codex 같은 코딩 에이전트가 셸을 통해 바로 호출할 수 있는 밈 저장소. 한 번의 `meme fetch`로 온라인에서 밈을 가져오고, `meme search`로 의미 기반 검색까지 한 번에 된다.

---

## AI-first 설계 규칙

- **기본은 JSON**. 터미널 예쁨은 opt-in (`--pretty`).
- **stdout은 결과만**, 진행/경고/에러는 **stderr**. 파이프 체이닝 안전.
- **구조화된 에러 분류**: `rate_limited` / `payment_required` / `unauthorized` / `not_found` / `timeout` / `request_failed` — 에이전트가 코드로 분기 가능.
- **시크릿 자동 마스킹**: API 키, UUID, Bearer 토큰은 출력에서 가려짐.
- **중복 감지 자동**: `(provider, source_id)` + SHA256 콘텐츠 해시. 재호출 시 API 낭비 없음.
- **이미지 생성만 전용 API**, 나머지 AI 작업(이미지 분석, 메타데이터 추출)은 `claude --print` subprocess로 위임.

---

## 설치

```bash
uv sync
cp .env.sample .env   # API 키 채우기 (선택)
uv run meme --help
```

Python 3.11+ 필요. `uv tool install .` 하면 `meme` 명령이 전역으로 설치된다.

---

## 세 가지 미디어 입수 경로

모두 **동일한 저장 흐름** (해시 중복 체크 → AI 메타 추출 → DB + 벡터 upsert) 을 통과한다.

### 1. `add` — 로컬 파일 업로드

```bash
meme add ./pepe.png
meme add ./pepe.png --name "페페" --tags "frog,sad" --force
```

### 2. `generate` — AI로 이미지 생성

```bash
meme generate "수학 풀다 지친 강아지" -p gemini
meme generate "surprised pikachu" -p replicate --model flux-dev --no-save
```

지원 프로바이더: `gemini`, `replicate`, `leonardo`.

### 3. `fetch` — 온라인 무료 API에서 가져오기

```bash
meme fetch "surprised pikachu" -p giphy --limit 5
meme fetch "cute cat"          -p openverse     # API 키 불필요
meme fetch "happy dog"         -p reddit --subreddit aww
```

지원 프로바이더: `giphy`, `unsplash`, `pexels`, `pixabay`, `openverse`, `reddit`.

---

## 검색

```bash
meme search "페페"                           # FTS5 trigram (키워드)
meme search "슬픈 개구리" --semantic         # bge-m3 벡터 유사도
meme search "cat" --source fetched --from-provider giphy
```

- 3자 이상은 trigram FTS5, 미만은 LIKE 폴백.
- `--semantic`은 ChromaDB + `BAAI/bge-m3` 임베딩 (최초 1회 모델 다운로드).

---

## 글로벌 옵션

| 플래그 | 환경변수 | 설명 |
|---|---|---|
| `--pretty` | `MEME_CLI_PRETTY` | 사람용 Rich 출력 활성화 |
| `--timeout N` | `MEME_CLI_TIMEOUT` | HTTP 타임아웃 (초, 기본 120) |
| `--yes` / `-y` | `MEME_CLI_ASSUME_YES` | 확인 프롬프트 자동 승인 |
| — | `MEME_CLI_DIR` | 저장소 루트 (기본 `~/.meme-cli`) |

글로벌 옵션은 **서브커맨드 앞에** 둔다: `meme --pretty search "pepe"`.

---

## 전체 커맨드

| 커맨드 | 설명 |
|---|---|
| `add <file>` | 로컬 파일 업로드 + AI 메타 자동 생성 |
| `generate <prompt>` | AI 이미지 생성 (자동 저장, `--no-save`로 스킵) |
| `fetch <query> -p <provider>` | 온라인 API에서 검색·다운로드·저장 |
| `search <query>` | 키워드/시맨틱 검색 |
| `list` | 전체 목록 |
| `get <id>` | 상세 조회 |
| `edit <id>` | 메타데이터 수정 (AI 재분석도 가능) |
| `remove <id>` | 삭제 |
| `balance -p <provider>` | 생성 프로바이더 크레딧 조회 |
| `providers` | 등록된 프로바이더 + 인증 상태 |
| `models -p <provider>` | 지원 모델 별명 |
| `info` | 저장소 경로·상태 덤프 |

---

## 환경변수

### 생성 프로바이더

| 키 | 용도 |
|---|---|
| `GEMINI_API_KEYS` | Gemini (쉼표 여러 키, 429 시 자동 전환) |
| `REPLICATE_API_TOKEN` | Replicate (FLUX / SD / Imagen 등) |
| `LEONARDO_API_KEY` | Leonardo.ai (Phoenix / FLUX 등) |

### 가져오기 프로바이더

| 키 | 용도 |
|---|---|
| `GIPHY_API_KEY` | Giphy (GIF 밈 핵심 소스) |
| `UNSPLASH_ACCESS_KEY` | Unsplash (고품질 사진) |
| `PEXELS_API_KEY` | Pexels (사진/비디오) |
| `PIXABAY_API_KEY` | Pixabay (사진/일러스트/벡터/비디오) |
| — | Openverse는 키 불필요 |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit OAuth (선택, 분당 100회). 없으면 무인증 10회/분 |
| `REDDIT_USER_AGENT` / `REDDIT_DEFAULT_SUBREDDIT` | Reddit 추가 설정 (선택) |

`.env` 파일은 자동 로드된다 (python-dotenv).

---

## 저장소 구조

```
~/.meme-cli/
├── memes.db    # SQLite + FTS5 (PRAGMA user_version으로 마이그레이션)
├── media/      # 원본 바이너리 (UUID 파일명)
└── chroma/     # 벡터 임베딩 (bge-m3)
```

`MEME_CLI_DIR`로 경로 오버라이드 가능.

---

## 아키텍처 한눈에

```
┌─ add ────┐
├─ generate┼─► saver.save_media() ──► hash dedup ──► AI 메타 ──► store.save + embedding upsert
└─ fetch ──┘                                       (claude --print)
```

- `store.py` — SQLite + FTS5, `PRAGMA user_version` 기반 선형 마이그레이션
- `embeddings.py` — ChromaDB + bge-m3 (lazy load)
- `saver.py` — 3가지 입수 경로의 단일 통합 진입점
- `ai.py` — 비-생성 AI의 유일한 경로 (`claude --print` subprocess)
- `output.py` — `--pretty` 라우팅, 시크릿 마스킹, 에러 분류
- `providers/` — 생성 (`ImageProvider`) / 가져오기 (`FetchProvider`) ABC 두 종

---

## 개발

```bash
pytest tests/                                      # 전체 테스트
pytest tests/test_store.py::test_save_and_get -v   # 단일 테스트
```

테스트는 `conftest.py:tmp_store_dir` fixture로 DB/미디어 디렉토리 격리. EmbeddingStore와 `claude --print`는 mock 처리. Fetch 프로바이더는 `respx` + `httpx.MockTransport`로 네트워크 없이 검증.

자세한 내부 규칙과 컨벤션은 [`CLAUDE.md`](./CLAUDE.md) 참고.
