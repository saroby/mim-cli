"""AI 친화적 출력 유틸리티.

- `--json` 모드: stdout에는 JSON만, stderr에는 진행 메시지/에러 상세.
- 인간 모드: Rich 박스/컬러로 stdout에 예쁘게 출력.
- 시크릿 마스킹: API 키 패턴을 에러 메시지에서 자동 제거.
"""

from __future__ import annotations

import json as _json
import os
import re
import sys
from typing import Any, NoReturn

import typer
from rich.console import Console

# stderr 출력용 콘솔 (진행/로그/에러 메시지)
err_console = Console(stderr=True)
# stdout 출력용 콘솔 (인간 모드에서만 데이터 렌더링)
out_console = Console()


# 글로벌 플래그 (main callback에서 설정).
# AI 친화 기본값: JSON 출력. 사람이 쓸 때는 --pretty로 Rich UI 활성화.
_state: dict[str, Any] = {
    "pretty": False,
    "timeout": 120.0,
    "assume_yes": False,
}


def set_flags(*, pretty: bool, timeout: float, assume_yes: bool) -> None:
    _state["pretty"] = pretty
    _state["timeout"] = timeout
    _state["assume_yes"] = assume_yes


def is_json() -> bool:
    """JSON 모드 여부 (--pretty가 아닐 때 = 기본값)."""
    return not bool(_state["pretty"])


def is_pretty() -> bool:
    return bool(_state["pretty"])


def get_timeout() -> float:
    return float(_state["timeout"])


def is_assume_yes() -> bool:
    return bool(_state["assume_yes"])


# ────────────────────────────────────────────────────────────
# 시크릿 마스킹
# ────────────────────────────────────────────────────────────

# Gemini(AIzaSy*), Replicate(r8_*), Leonardo(UUID), Bearer 토큰 패턴.
# + URL 쿼리스트링의 api_key=, key= 파라미터 (Giphy/Pixabay 스타일).
# + 헤더류: Authorization: / Client-ID / X-Api-Key.
_SECRET_PATTERNS = [
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{30,}"),
    re.compile(r"r8_[A-Za-z0-9]{30,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]+"),
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
    # URL 쿼리스트링 (Giphy/Pixabay): ?api_key=XXX 또는 &key=XXX
    re.compile(r"(?i)([?&](?:api[_-]?key|access[_-]?key|key|token)=)[^&\s\"']+"),
    # 헤더류: "Client-ID abc", "X-Api-Key: abc"
    re.compile(r"(?i)(Client-ID\s+)[A-Za-z0-9_\-]+"),
    re.compile(r"(?i)(X-Api-Key[:=]\s*)[A-Za-z0-9_\-]+"),
]


def _register_runtime_secrets() -> list[re.Pattern[str]]:
    """환경변수에 실제 설정된 시크릿 값을 런타임에 수집해 리터럴 매칭 패턴으로 등록.

    패턴으로 모든 형태의 API 키를 잡는 건 현실적으로 어렵기 때문에
    (예: Giphy 키는 고유 접두어 없는 32자 영숫자) 실제 로드된 값 자체를
    마스킹 대상에 추가.
    """
    env_keys = (
        "GIPHY_API_KEY", "UNSPLASH_ACCESS_KEY", "PEXELS_API_KEY",
        "PIXABAY_API_KEY", "GEMINI_API_KEYS", "REPLICATE_API_TOKEN",
        "REPLICATE_API_KEY", "LEONARDO_API_KEY", "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
    )
    out: list[re.Pattern[str]] = []
    for name in env_keys:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        # 쉼표 구분 다중 키(GEMINI_API_KEYS) 지원
        for part in raw.split(","):
            val = part.strip()
            if len(val) >= 8:  # 너무 짧은 값은 과매칭 위험
                out.append(re.compile(re.escape(val)))
    return out


def mask_secret(text: str) -> str:
    """API 키/토큰/UUID + 런타임 환경변수 값을 마스킹."""
    if not text:
        return text
    result = text
    # 키/값 형태 (쿼리스트링, 헤더 prefix) — 접두어 보존 후 값만 치환
    for pat in _SECRET_PATTERNS[4:7]:
        result = pat.sub(lambda m: f"{m.group(1)}***", result)
    # 토큰 자체 패턴
    for pat in _SECRET_PATTERNS[:4]:
        result = pat.sub(lambda m: _mask_match(m.group(0)), result)
    # 실제 로드된 환경변수 값 리터럴 치환
    for pat in _register_runtime_secrets():
        result = pat.sub("***", result)
    return result


def _mask_match(s: str) -> str:
    if len(s) <= 12:
        return "***"
    return f"{s[:6]}***{s[-4:]}"


# ────────────────────────────────────────────────────────────
# 출력 헬퍼
# ────────────────────────────────────────────────────────────


def emit(data: Any, *, human_render=None) -> None:
    """데이터를 출력. JSON 모드면 stdout에 JSON, 아니면 human_render() 호출.

    human_render: 인자 없는 callable. None이면 기본 pretty-print.
    """
    if is_json():
        sys.stdout.write(_json.dumps(data, ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    if human_render is not None:
        human_render()
    else:
        out_console.print(data)


def emit_error(error_type: str, message: str, *, exit_code: int = 1, **extra: Any) -> NoReturn:
    """에러 출력 후 종료. JSON 모드면 stderr에 JSON, 아니면 빨간 메시지."""
    masked = mask_secret(str(message))
    if is_json():
        payload = {"error": error_type, "message": masked, **extra}
        sys.stderr.write(_json.dumps(payload, ensure_ascii=False))
        sys.stderr.write("\n")
        sys.stderr.flush()
    else:
        err_console.print(f"[red]{error_type}: {masked}[/red]")
        for k, v in extra.items():
            err_console.print(f"[dim]  {k}: {v}[/dim]")
    raise typer.Exit(exit_code)


def log(msg: str) -> None:
    """진행/상태 메시지. 항상 stderr로. JSON 모드에서는 생략 (AI 토큰 낭비 방지)."""
    if is_json():
        return
    err_console.print(f"[dim]{msg}[/dim]")


def confirm(prompt: str, *, default: bool = False, destructive: bool = False) -> bool:
    """사용자 확인.

    - `--yes`/`MIM_CLI_ASSUME_YES`면 자동 승인.
    - pretty(사람) 모드면 typer 프롬프트.
    - JSON 모드면: destructive=True일 때 --yes 없으면 에러로 실패.
      그 외에는 자동 승인 (비파괴적 결정은 스크립트에 맡김).
    """
    if is_assume_yes():
        return True
    if is_json():
        if destructive:
            emit_error(
                "confirmation_required",
                "파괴적 작업은 비대화형(JSON) 모드에서 --yes 또는 MIM_CLI_ASSUME_YES가 필요합니다.",
            )
        return True
    return typer.confirm(prompt, default=default)


def classify_error(e: Exception) -> tuple[str, dict[str, Any]]:
    """예외를 구조화된 에러 타입과 메타데이터로 분류."""
    msg = str(e).lower()
    extra: dict[str, Any] = {}
    if (
        "429" in msg or "resource_exhausted" in msg or "quota" in msg
        or "rate limit" in msg or "too many requests" in msg
    ):
        return "rate_limited", extra
    if "402" in msg or "payment" in msg or "billing" in msg:
        return "payment_required", extra
    if (
        "401" in msg or "403" in msg or "unauthorized" in msg
        or "forbidden" in msg or "authentication" in msg or "invalid api key" in msg
    ):
        return "unauthorized", extra
    if "404" in msg or "not found" in msg:
        return "not_found", extra
    if "timeout" in msg or "timed out" in msg:
        return "timeout", extra
    return "request_failed", extra
