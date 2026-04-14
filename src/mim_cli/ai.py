import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GeneratedMetadata:
    name: str
    description: str
    tags: list[str]
    emotions: list[str]
    context: list[str]


PROMPT_TEMPLATE = """{path}

위 파일을 분석해서 밈 라이브러리 검색에 최적화된 메타데이터를 JSON만 반환해 (다른 텍스트 없이):
{{
  "name": "한국어 짧은 이름 (10자 이내)",
  "description": "무슨 장면인지 상세 설명 (한국어, 2-3문장)",
  "tags": ["한국어 검색 태그 5-10개 (예: 충격, 폭발, 웃음, 피카츄)"],
  "emotions": ["한국어 감정 태그 (예: 충격, 기쁨, 슬픔, 분노, 당혹)"],
  "context": ["한국어 맥락 태그 (예: 반응, 전환, 인트로, 액션, 반전)"]
}}"""


class MetadataGenerator:
    def generate(self, file_path: Path) -> GeneratedMetadata:
        """claude --print로 파일을 분석해 메타데이터 생성.

        file_path는 저장소 내부의 UUID 경로를 기대 (saver.py가 dest를 전달).
        방어적으로 resolve()로 정규화하여 프롬프트 주입 가능한 문자
        (개행/백틱 등)가 포함된 외부 경로가 섞여도 안전하게.
        """
        safe = Path(file_path).resolve()
        prompt = PROMPT_TEMPLATE.format(path=safe)
        result = subprocess.run(
            ["claude", "--model", "claude-haiku-4-5-20251001", "--print", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude 실행 실패: {result.stderr[:200]}")
        return self._parse(result.stdout)

    def _parse(self, text: str) -> GeneratedMetadata:
        """JSON 블록 추출 — 마크다운 코드 블록 포함 처리.

        중괄호 균형을 추적해 최외곽 객체를 찾음 (정규식 catastrophic
        backtracking 회피).
        """
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        block = _find_first_json_object(text)
        if block is None:
            raise ValueError(f"JSON 파싱 실패. Claude 응답:\n{text[:300]}")
        data = json.loads(block)
        return GeneratedMetadata(
            name=data["name"],
            description=data["description"],
            tags=data.get("tags", []),
            emotions=data.get("emotions", []),
            context=data.get("context", []),
        )


def _find_first_json_object(text: str) -> str | None:
    """중괄호 균형 스캔으로 최초의 JSON 오브젝트 블록 반환.

    문자열 리터럴 내부의 중괄호/이스케이프를 구분해서 추적.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
