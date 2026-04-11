import inspect
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import typer
from rich.panel import Panel
from rich.table import Table

from meme_cli.config import get_db_path, get_media_dir, get_chroma_dir
from meme_cli.embeddings import EmbeddingStore
from meme_cli.models import MediaItem
from meme_cli.output import (
    classify_error,
    confirm,
    emit,
    emit_error,
    err_console,
    get_timeout,
    is_json,
    is_pretty,
    log,
    mask_secret,
    out_console,
    set_flags,
)
from meme_cli.providers import FetchProvider, ImageProvider
from meme_cli.saver import MetaOverride, save_media
from meme_cli.search import MediaSearch
from meme_cli.store import MediaStore

GEN_PROVIDERS: dict[str, type[ImageProvider]] = {}
FETCH_PROVIDERS: dict[str, type[FetchProvider]] = {}


def _register_providers() -> None:
    from meme_cli.providers.gemini import GeminiProvider
    from meme_cli.providers.leonardo import LeonardoProvider
    from meme_cli.providers.replicate import ReplicateProvider
    from meme_cli.providers.fetch.giphy import GiphyProvider
    from meme_cli.providers.fetch.openverse import OpenverseProvider
    from meme_cli.providers.fetch.pexels import PexelsProvider
    from meme_cli.providers.fetch.pixabay import PixabayProvider
    from meme_cli.providers.fetch.reddit import RedditProvider
    from meme_cli.providers.fetch.unsplash import UnsplashProvider

    GEN_PROVIDERS["gemini"] = GeminiProvider
    GEN_PROVIDERS["replicate"] = ReplicateProvider
    GEN_PROVIDERS["leonardo"] = LeonardoProvider
    FETCH_PROVIDERS["giphy"] = GiphyProvider
    FETCH_PROVIDERS["reddit"] = RedditProvider
    FETCH_PROVIDERS["unsplash"] = UnsplashProvider
    FETCH_PROVIDERS["pexels"] = PexelsProvider
    FETCH_PROVIDERS["pixabay"] = PixabayProvider
    FETCH_PROVIDERS["openverse"] = OpenverseProvider


_register_providers()

PROVIDER_ENV = {
    "gemini": "GEMINI_API_KEYS",
    "replicate": "REPLICATE_API_TOKEN",
    "leonardo": "LEONARDO_API_KEY",
    "giphy": "GIPHY_API_KEY",
    "unsplash": "UNSPLASH_ACCESS_KEY",
    "pexels": "PEXELS_API_KEY",
    "pixabay": "PIXABAY_API_KEY",
    "openverse": None,  # 키 불필요
    "reddit": None,     # 무인증 기본. OAuth는 REDDIT_CLIENT_ID/SECRET 선택
}

app = typer.Typer(
    help="밈 미디어 관리 CLI (AI 친화 기본값: JSON 출력. 사람용은 --pretty)",
    no_args_is_help=True,
)


@app.callback()
def main(
    pretty: bool = typer.Option(
        False, "--pretty", envvar="MEME_CLI_PRETTY",
        help="사람용 Rich 출력. 미지정 시 기본은 JSON (AI 친화)",
    ),
    timeout: float = typer.Option(
        120.0, "--timeout", envvar="MEME_CLI_TIMEOUT",
        help="HTTP 요청/폴링 타임아웃 (초)",
    ),
    assume_yes: bool = typer.Option(
        False, "--yes", "-y", envvar="MEME_CLI_ASSUME_YES",
        help="확인 프롬프트 자동 승인",
    ),
) -> None:
    """글로벌 옵션."""
    load_dotenv()
    set_flags(pretty=pretty, timeout=timeout, assume_yes=assume_yes)


def _get_store() -> MediaStore:
    return MediaStore(db_path=get_db_path())


def _get_emb_store() -> EmbeddingStore:
    return EmbeddingStore(chroma_dir=get_chroma_dir())


def _detect_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        return "video"
    if suffix == ".gif":
        return "gif"
    return "image"


def _resolve_item(store: MediaStore, item_id: str) -> MediaItem:
    item = store.get(item_id)
    if item:
        return item
    matches = [i for i in store.list_all() if i.id.startswith(item_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        emit_error("ambiguous_id", f"ID 접두어 '{item_id}'에 여러 항목이 일치합니다.")
    emit_error("not_found", f"ID '{item_id}'를 찾을 수 없습니다.")


def _build_gen_provider(name: str, model: str | None = None) -> ImageProvider:
    if name not in GEN_PROVIDERS:
        available = ", ".join(GEN_PROVIDERS.keys())
        emit_error("unknown_provider", f"알 수 없는 생성 프로바이더: {name}", available=available)
    cls = GEN_PROVIDERS[name]
    kwargs: dict = {"timeout": get_timeout()}
    if model:
        kwargs["model"] = model
    return cls(**kwargs)


def _build_fetch_provider(name: str, **extra) -> FetchProvider:
    """가져오기 프로바이더 빌드. extra kwargs는 해당 클래스 생성자가 받는 것만 전달."""
    if name not in FETCH_PROVIDERS:
        available = ", ".join(FETCH_PROVIDERS.keys())
        emit_error("unknown_provider", f"알 수 없는 가져오기 프로바이더: {name}", available=available)
    cls = FETCH_PROVIDERS[name]
    kwargs: dict = {"timeout": get_timeout()}
    sig = inspect.signature(cls.__init__)
    for k, v in extra.items():
        if v is not None and k in sig.parameters:
            kwargs[k] = v
    return cls(**kwargs)


def _suffix_from_mime(mime: str) -> str:
    """MIME → 파일 확장자."""
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
    }
    return mapping.get(mime.lower(), ".bin")


def _media_type_from_mime(mime: str) -> str:
    """MIME → media_type."""
    m = mime.lower()
    if m.startswith("video/"):
        return "video"
    if m == "image/gif":
        return "gif"
    return "image"


@app.command()
def add(
    file: Path = typer.Argument(..., help="추가할 미디어 파일 경로"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="이름 (미지정 시 AI 분석)"),
    description: Optional[str] = typer.Option(None, "--desc", help="설명 (미지정 시 AI 분석)"),
    tags: Optional[str] = typer.Option(None, "--tags", help="태그 (쉼표 구분, 미지정 시 AI 분석)"),
    emotions: Optional[str] = typer.Option(None, "--emotions", help="감정 (쉼표 구분, 미지정 시 AI 분석)"),
    context: Optional[str] = typer.Option(None, "--context", help="맥락 (쉼표 구분, 미지정 시 AI 분석)"),
    skip_metadata: bool = typer.Option(False, "--skip-metadata", help="AI 이미지 분석 완전 스킵 (빈 값으로 저장)"),
    force: bool = typer.Option(False, "--force", help="해시 중복 감지 무시하고 강제 저장"),
) -> None:
    """미디어 파일을 저장소에 추가. 메타 옵션 전부 제공 시 AI 스킵, 부분 제공 시 비어있는 필드만 AI가 채움."""
    if not file.exists():
        emit_error("file_not_found", f"파일을 찾을 수 없습니다: {file}")

    override = MetaOverride(
        name=name,
        description=description,
        tags=_split_csv(tags),
        emotions=_split_csv(emotions),
        context=_split_csv(context),
    )

    if not skip_metadata and not override.is_complete():
        log("AI 이미지 분석 중... (claude --print)")

    store = _get_store()
    emb_store = _get_emb_store()
    item, is_new = save_media(
        store=store,
        emb_store=emb_store,
        data=file,
        suffix=file.suffix,
        media_type=_detect_type(file),
        source="upload",
        meta_override=override,
        skip_metadata=skip_metadata,
        force=force,
    )

    def render():
        if not is_new:
            out_console.print(Panel(
                f"[yellow]이미 존재하는 미디어[/yellow]\n"
                f"ID: {item.id}\n"
                f"이름: {item.name}\n"
                f"[dim](--force로 강제 저장 가능)[/dim]",
                title="중복 감지",
            ))
            return
        out_console.print(Panel(
            f"[bold green]추가됨[/bold green]\n"
            f"ID: {item.id}\n"
            f"이름: {item.name}\n"
            f"태그: {', '.join(item.tags)}\n"
            f"감정: {', '.join(item.emotions)}\n"
            f"맥락: {', '.join(item.context)}",
            title="밈 추가 완료",
        ))

    payload = item.to_dict()
    payload["skipped"] = not is_new
    if not is_new:
        payload["existing_id"] = item.id
    emit(payload, human_render=render)


def _split_csv(s: Optional[str]) -> Optional[list[str]]:
    """쉼표 구분 문자열을 list로. None이면 None 유지 (AI가 채움)."""
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


@app.command()
def search(
    query: str = typer.Argument(..., help="검색 쿼리"),
    media_type: Optional[str] = typer.Option(None, "--type", "-t", help="타입 필터: video|gif|image"),
    limit: int = typer.Option(20, "--limit", "-l", help="최대 결과 수"),
    semantic: bool = typer.Option(False, "--semantic", "-s", help="유사 의미 검색 (벡터)"),
    source: Optional[str] = typer.Option(None, "--source", help="출처 필터: upload | generated | fetched"),
    from_provider: Optional[str] = typer.Option(None, "--from-provider", help="특정 프로바이더 출처 필터 (giphy, leonardo 등)"),
) -> None:
    """메타데이터 기반 밈 검색."""
    store = _get_store()

    if semantic:
        emb_store = _get_emb_store()
        searcher = MediaSearch(store, embedding_store=emb_store)
        results = searcher.semantic_query(
            query, media_type=media_type, limit=limit,
            source=source, source_provider=from_provider,
        )
    else:
        searcher = MediaSearch(store)
        results = searcher.query(
            query, media_type=media_type, limit=limit,
            source=source, source_provider=from_provider,
        )

    def render():
        if not results:
            err_console.print("[yellow]결과 없음[/yellow]")
            return
        mode_label = "[cyan]의미 검색[/cyan]" if semantic else "키워드 검색"
        table = Table(title=f"{mode_label}: '{query}'")
        table.add_column("ID", style="dim", width=8)
        table.add_column("이름", style="bold")
        table.add_column("타입")
        table.add_column("태그")
        table.add_column("감정")
        for r in results:
            table.add_row(
                r.id[:8], r.name, r.media_type,
                ", ".join(r.tags[:4]),
                ", ".join(r.emotions[:3]),
            )
        out_console.print(table)

    emit([r.to_dict() for r in results], human_render=render)


@app.command()
def edit(
    item_id: str = typer.Argument(..., help="수정할 아이템 ID"),
    name: Optional[str] = typer.Option(None, "--name", help="새 이름"),
    description: Optional[str] = typer.Option(None, "--desc", help="새 설명"),
    tags: Optional[str] = typer.Option(None, "--tags", help="새 태그 (쉼표 구분)"),
    emotions: Optional[str] = typer.Option(None, "--emotions", help="새 감정 태그"),
    context: Optional[str] = typer.Option(None, "--context", help="새 맥락 태그"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="대화형 수정 (JSON 모드에선 비활성)"),
) -> None:
    """메타데이터 수정."""
    store = _get_store()
    item = _resolve_item(store, item_id)

    if interactive and not is_json():
        out_console.print(Panel(
            f"이름: {item.name}\n태그: {', '.join(item.tags)}\n"
            f"감정: {', '.join(item.emotions)}\n맥락: {', '.join(item.context)}",
            title="현재 메타데이터",
        ))
        name = typer.prompt("이름", default=item.name)
        description = typer.prompt("설명", default=item.description)
        tags = typer.prompt("태그 (쉼표 구분)", default=",".join(item.tags))
        emotions = typer.prompt("감정 (쉼표 구분)", default=",".join(item.emotions))
        context = typer.prompt("맥락 (쉼표 구분)", default=",".join(item.context))

    if name is not None:
        item.name = name
    if description is not None:
        item.description = description
    if tags is not None:
        item.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if emotions is not None:
        item.emotions = [e.strip() for e in emotions.split(",") if e.strip()]
    if context is not None:
        item.context = [c.strip() for c in context.split(",") if c.strip()]

    item.touch()
    store.update(item)
    emb_store = _get_emb_store()
    emb_store.upsert(item)

    def render():
        out_console.print(f"[green]수정됨: {item.name} ({item.id[:8]})[/green]")

    emit(item.to_dict(), human_render=render)


@app.command(name="list")
def list_items(
    media_type: Optional[str] = typer.Option(None, "--type", "-t", help="타입 필터"),
) -> None:
    """저장된 모든 밈 목록."""
    store = _get_store()
    items = store.list_all(media_type=media_type)

    def render():
        if not items:
            err_console.print("[yellow]저장된 밈이 없습니다. 'meme add <파일>'로 추가하세요.[/yellow]")
            return
        table = Table(title="밈 목록")
        table.add_column("ID", style="dim", width=8)
        table.add_column("이름", style="bold")
        table.add_column("타입")
        table.add_column("태그")
        table.add_column("감정")
        table.add_column("날짜")
        for i in items:
            table.add_row(
                i.id[:8], i.name, i.media_type,
                ", ".join(i.tags[:3]),
                ", ".join(i.emotions[:2]),
                i.created_at[:10],
            )
        out_console.print(table)

    emit([i.to_dict() for i in items], human_render=render)


@app.command()
def get(
    item_id: str = typer.Argument(..., help="아이템 ID"),
) -> None:
    """특정 밈의 상세 정보."""
    store = _get_store()
    item = _resolve_item(store, item_id)

    def render():
        out_console.print(Panel(
            f"[bold]ID:[/bold] {item.id}\n"
            f"[bold]이름:[/bold] {item.name}\n"
            f"[bold]타입:[/bold] {item.media_type}\n"
            f"[bold]경로:[/bold] {item.path}\n"
            f"[bold]설명:[/bold] {item.description}\n"
            f"[bold]태그:[/bold] {', '.join(item.tags)}\n"
            f"[bold]감정:[/bold] {', '.join(item.emotions)}\n"
            f"[bold]맥락:[/bold] {', '.join(item.context)}\n"
            f"[bold]추가일:[/bold] {item.created_at[:10]}",
            title=item.name,
        ))

    emit(item.to_dict(), human_render=render)


@app.command()
def remove(
    item_id: str = typer.Argument(..., help="삭제할 아이템 ID"),
    keep_file: bool = typer.Option(False, "--keep-file", help="원본 파일 유지"),
) -> None:
    """저장소에서 밈 삭제. 글로벌 --yes 또는 MEME_CLI_ASSUME_YES로 확인 건너뛰기."""
    store = _get_store()
    item = _resolve_item(store, item_id)

    if not confirm(f"'{item.name}'을 삭제하시겠습니까?", destructive=True):
        raise typer.Abort()

    if not keep_file:
        file_path = Path(item.path)
        if file_path.exists():
            file_path.unlink()

    store.delete(item.id)
    emb_store = _get_emb_store()
    emb_store.delete(item.id)

    def render():
        out_console.print(f"[green]삭제됨: {item.name}[/green]")

    emit({"deleted": True, "id": item.id, "name": item.name}, human_render=render)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="생성할 이미지 설명 (프롬프트)"),
    provider: str = typer.Option("gemini", "--provider", "-p", help="프로바이더 (gemini, replicate, leonardo)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="모델 별명 (meme models <provider>로 목록 확인)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="외부 출력 경로 (지정해도 저장소에는 별도 사본 보관, --no-save면 외부에만)"),
    aspect_ratio: Optional[str] = typer.Option(None, "--aspect-ratio", "-ar", help="비율 (1:1, 16:9, 9:16, 4:3, 3:4)"),
    num_images: int = typer.Option(1, "--num-images", "-n", min=1, max=10, help="생성 개수 (AI 메타데이터는 N번 호출)"),
    save: bool = typer.Option(True, "--save/--no-save", help="저장소에 자동 등록 (기본 on). --no-save면 파일만 생성"),
    skip_metadata: bool = typer.Option(False, "--skip-metadata", help="저장 시 AI 이미지 분석 스킵 (프롬프트/모델 정보만)"),
    open_file: bool = typer.Option(
        False, "--open/--no-open",
        help="생성 후 파일 열기. --pretty + TTY 환경에서만 동작",
    ),
) -> None:
    """AI로 이미지 생성. 기본값: 저장소 자동 저장. --no-save로 DB 스킵."""
    gen_provider = _build_gen_provider(provider, model=model)

    if not gen_provider.check_auth():
        env_var = PROVIDER_ENV.get(provider, "API 키")
        emit_error(
            "unauthorized",
            f"{provider} API 키가 설정되지 않았습니다.",
            env_var=env_var,
        )

    tmp_dir = get_media_dir() / "generated"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    store = _get_store() if save else None
    emb_store = _get_emb_store() if save else None

    generated: list[dict] = []
    for idx in range(num_images):
        # 1차 파일 쓰기 위치: --output이 주어지면 거기 + saver가 저장소 내부에도 복사
        #                   아니면 임시 파일만
        target = _pick_generate_target(output, num_images, idx, tmp_dir)

        log(f"[{idx + 1}/{num_images}] {provider}({gen_provider.name})로 생성 중...")
        try:
            result = gen_provider.generate(
                prompt=prompt,
                output_path=target,
                aspect_ratio=aspect_ratio,
            )
        except Exception as e:
            err_type, extra = classify_error(e)
            emit_error(err_type, mask_secret(str(e)), provider=provider, **extra)

        record: dict = {
            "path": str(result.path),
            "prompt": result.prompt,
            "provider": result.provider,
            "model": result.model,
        }

        if save:
            # 저장소에 등록 (해시 중복 체크 포함)
            if not skip_metadata:
                log("AI 이미지 분석 중... (claude --print)")
            item, is_new = save_media(
                store=store, emb_store=emb_store,
                data=Path(result.path),
                suffix=Path(result.path).suffix,
                media_type="image",
                source="generated",
                source_provider=result.provider,
                model=result.model,
                prompt=result.prompt,
                skip_metadata=skip_metadata,
            )
            record["id"] = item.id
            record["skipped"] = not is_new
            if not is_new:
                record["existing_id"] = item.id

        generated.append(record)

    def render():
        for g in generated:
            title = "이미지 생성"
            body = (
                f"[bold green]생성 완료[/bold green]\n"
                f"프롬프트: {g['prompt']}\n"
                f"프로바이더: {g['provider']} ({g['model']})\n"
                f"저장 경로: {g['path']}"
            )
            if g.get("id"):
                body += f"\nID: {g['id']}"
            if g.get("skipped"):
                title = "[yellow]중복 감지[/yellow]"
            out_console.print(Panel(body, title=title))

    emit(generated[0] if num_images == 1 else generated, human_render=render)

    if open_file and is_pretty() and sys.stdout.isatty():
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        for g in generated:
            subprocess.Popen([opener, g["path"]])


def _pick_generate_target(
    output: Optional[Path],
    num_images: int,
    idx: int,
    tmp_dir: Path,
) -> Path:
    """generate의 실제 파일 쓰기 경로 결정."""
    if output and num_images == 1:
        return output
    if output and num_images > 1:
        return output.with_stem(f"{output.stem}_{idx + 1}")
    return tmp_dir / f"{uuid.uuid4()}.png"


@app.command()
def balance(
    provider: str = typer.Argument(..., help="프로바이더 (leonardo, gemini, replicate)"),
) -> None:
    """프로바이더 계정 잔여 크레딧/토큰 조회 (표준화된 스키마). 생성 프로바이더만 지원."""
    gen = _build_gen_provider(provider)

    if not gen.check_auth():
        emit_error(
            "unauthorized",
            f"{provider} API 키가 설정되지 않았습니다.",
            env_var=PROVIDER_ENV.get(provider),
        )

    try:
        info = gen.balance()
    except NotImplementedError as e:
        emit_error("not_supported", str(e), provider=provider)
    except Exception as e:
        err_type, extra = classify_error(e)
        emit_error(err_type, mask_secret(str(e)), provider=provider, **extra)

    def render():
        table = Table(title=f"{provider} 잔여", show_header=False)
        table.add_column("항목", style="cyan")
        table.add_column("값")
        for key in ("provider", "available", "plan", "quota_remaining", "quota_total", "unit", "renews_at", "note"):
            val = info.get(key)
            if val is None:
                continue
            table.add_row(key, str(val))
        details = info.get("details") or {}
        for k, v in details.items():
            if v is None:
                continue
            if isinstance(v, list):
                for i, entry in enumerate(v):
                    if isinstance(entry, dict):
                        table.add_row(f"{k}[{i}]", " | ".join(f"{a}={b}" for a, b in entry.items()))
                    else:
                        table.add_row(f"{k}[{i}]", str(entry))
            else:
                table.add_row(k, str(v))
        out_console.print(table)

    emit(info, human_render=render)


@app.command()
def providers(
    kind: Optional[str] = typer.Option(None, "--type", "-t", help="필터: generate | fetch"),
) -> None:
    """등록된 프로바이더 목록 + 인증 상태."""
    items: list[dict] = []

    def collect(registry: dict, k: str) -> None:
        if kind and kind != k:
            return
        for name, cls in registry.items():
            try:
                inst = cls(timeout=get_timeout())
                authed = inst.check_auth()
            except Exception:
                authed = False
            items.append({
                "name": name,
                "kind": k,
                "authenticated": authed,
                "env_var": PROVIDER_ENV.get(name),
            })

    collect(GEN_PROVIDERS, "generate")
    collect(FETCH_PROVIDERS, "fetch")

    def render():
        table = Table(title="프로바이더")
        table.add_column("이름", style="bold")
        table.add_column("종류")
        table.add_column("인증")
        table.add_column("환경변수")
        for p in items:
            authed_mark = "[green]✓[/green]" if p["authenticated"] else "[red]✗[/red]"
            table.add_row(p["name"], p["kind"], authed_mark, p["env_var"] or "-")
        out_console.print(table)

    emit(items, human_render=render)


@app.command()
def models(
    provider: str = typer.Argument(..., help="프로바이더 (leonardo, gemini, replicate)"),
) -> None:
    """프로바이더가 지원하는 모델 별명 목록. 생성 프로바이더 전용."""
    gen = _build_gen_provider(provider)
    items = gen.list_models()

    def render():
        table = Table(title=f"{provider} 모델")
        table.add_column("별명", style="bold cyan")
        table.add_column("ID")
        table.add_column("설명")
        for m in items:
            table.add_row(m.get("alias", ""), m.get("id", ""), m.get("note", "") or "")
        out_console.print(table)

    emit(items, human_render=render)


@app.command()
def info() -> None:
    """시스템 정보 (저장소 경로, 프로바이더 상태)."""
    store = _get_store()
    all_items = store.list_all()

    providers_info: list[dict] = []
    for name, cls in GEN_PROVIDERS.items():
        providers_info.append({
            "name": name, "kind": "generate",
            "authenticated": _safe_auth(cls),
            "env_var": PROVIDER_ENV.get(name),
        })
    for name, cls in FETCH_PROVIDERS.items():
        providers_info.append({
            "name": name, "kind": "fetch",
            "authenticated": _safe_auth(cls),
            "env_var": PROVIDER_ENV.get(name),
        })

    data = {
        "db_path": str(get_db_path()),
        "media_dir": str(get_media_dir()),
        "chroma_dir": str(get_chroma_dir()),
        "total_items": len(all_items),
        "providers": providers_info,
    }

    def render():
        table = Table(title="시스템 정보", show_header=False)
        table.add_column("항목", style="cyan")
        table.add_column("값")
        for k in ("db_path", "media_dir", "chroma_dir", "total_items"):
            table.add_row(k, str(data[k]))
        for p in data["providers"]:
            mark = "[green]✓[/green]" if p["authenticated"] else "[red]✗[/red]"
            table.add_row(f"provider:{p['name']}", f"{mark} ({p['env_var']})")
        out_console.print(table)

    emit(data, human_render=render)


def _safe_auth(cls) -> bool:
    try:
        return cls(timeout=get_timeout()).check_auth()
    except Exception:
        return False


@app.command()
def fetch(
    query: str = typer.Argument(..., help="검색 쿼리"),
    provider: str = typer.Option(..., "--provider", "-p", help="프로바이더 (giphy, reddit, unsplash, pexels, pixabay, openverse)"),
    limit: int = typer.Option(1, "--limit", "-l", min=1, max=50, help="가져올 개수"),
    media_type: Optional[str] = typer.Option(None, "--media-type", help="mime 필터 (image|gif|video). 프로바이더 지원 범위 내"),
    subreddit: Optional[str] = typer.Option(None, "--subreddit", help="(reddit 전용) 특정 서브레딧 내에서만 검색. 예: memes, reactiongifs"),
    save: bool = typer.Option(True, "--save/--no-save", help="저장소에 자동 등록 (기본 on). 중복은 조용히 스킵"),
    skip_metadata: bool = typer.Option(False, "--skip-metadata", help="AI 이미지 분석 스킵"),
) -> None:
    """온라인 무료 API에서 미디어 검색 + 자동 저장.

    중복(같은 source_id 또는 content_hash)은 조용히 스킵하고 기존 아이템 반환.
    쿼리는 프롬프트와 함께 저장되어 검색 가능.
    """
    fp = _build_fetch_provider(provider, subreddit=subreddit)

    if not fp.check_auth():
        emit_error(
            "unauthorized",
            f"{provider} API 키가 설정되지 않았습니다.",
            env_var=PROVIDER_ENV.get(provider),
        )

    log(f"{provider}에서 '{query}' 검색 중... (최대 {limit}개)")
    try:
        media_list = fp.search(query=query, limit=limit, media_type=media_type)
    except Exception as e:
        err_type, extra = classify_error(e)
        emit_error(err_type, mask_secret(str(e)), provider=provider, **extra)

    if not media_list:
        # 0건은 에러가 아님. 빈 배열 반환.
        emit([], human_render=lambda: err_console.print("[yellow]결과 없음[/yellow]"))
        return

    if not save:
        # 저장 안 하고 메타데이터만 반환
        payload = [
            {
                "source_url": m.source_url,
                "source_id": m.source_id,
                "mime_type": m.mime_type,
                "width": m.width,
                "height": m.height,
                "attribution": m.attribution,
                "license": m.license,
                "metadata": m.metadata,
            }
            for m in media_list
        ]

        def render():
            for p in payload:
                out_console.print(Panel(
                    f"[bold]source_id:[/bold] {p['source_id']}\n"
                    f"[bold]url:[/bold] {p['source_url']}\n"
                    f"[bold]mime:[/bold] {p['mime_type']}\n"
                    f"[bold]size:[/bold] {p.get('width')}x{p.get('height')}\n"
                    f"[bold]attribution:[/bold] {p.get('attribution') or '-'}\n"
                    f"[bold]license:[/bold] {p.get('license') or '-'}",
                    title=f"{provider} — {query}",
                ))

        emit(payload if len(payload) > 1 else payload[0], human_render=render)
        return

    store = _get_store()
    emb_store = _get_emb_store()

    records: list[dict] = []
    for i, m in enumerate(media_list):
        log(f"[{i + 1}/{len(media_list)}] 저장 중: source_id={m.source_id}")
        try:
            item, is_new = save_media(
                store=store, emb_store=emb_store,
                data=m.data,
                suffix=_suffix_from_mime(m.mime_type),
                media_type=_media_type_from_mime(m.mime_type),
                source="fetched",
                source_provider=fp.name,
                source_url=m.source_url,
                source_id=m.source_id,
                prompt=query,              # 검색 쿼리를 prompt로 저장
                attribution=m.attribution,
                license=m.license,
                license_url=m.license_url,
                width=m.width,
                height=m.height,
                skip_metadata=skip_metadata,
            )
        except Exception as e:
            err_type, extra = classify_error(e)
            # 한 건 실패해도 나머지는 계속
            records.append({
                "error": err_type,
                "source_id": m.source_id,
                "message": mask_secret(str(e)),
            })
            continue

        records.append({
            "id": item.id,
            "path": item.path,
            "name": item.name,
            "skipped": not is_new,
            **({"existing_id": item.id} if not is_new else {}),
            "source_url": m.source_url,
            "source_id": m.source_id,
            "provider": fp.name,
        })

    def render():
        for r in records:
            if r.get("error"):
                err_console.print(f"[red]{r['error']}: {r.get('source_id')}[/red]")
                continue
            title = "[yellow]중복 (스킵)[/yellow]" if r.get("skipped") else "[green]저장됨[/green]"
            out_console.print(Panel(
                f"ID: {r.get('id')}\n"
                f"이름: {r.get('name')}\n"
                f"출처: {r.get('provider')} / {r.get('source_id')}\n"
                f"URL: {r.get('source_url')}",
                title=title,
            ))

    emit(records if len(records) > 1 else records[0], human_render=render)
