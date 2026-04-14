import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from mim_cli.providers import GeneratedImage, ImageProvider


class GeminiProvider(ImageProvider):
    """Google Gemini 2.5 Flash 이미지 생성."""

    name = "gemini"
    DEFAULT_MODEL = "gemini-2.5-flash-image"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        raw = api_key or os.environ.get("GEMINI_API_KEYS", "")
        self._keys = [k.strip() for k in raw.split(",") if k.strip()]
        self._model = model or self.DEFAULT_MODEL
        self._clients: dict[str, genai.Client] = {}
        self._timeout = timeout

    def _get_client(self, key: str) -> genai.Client:
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key]

    def check_auth(self) -> bool:
        return len(self._keys) > 0

    def list_models(self) -> list[dict[str, Any]]:
        # Gemini는 별명 체계가 없어서 기본 모델만 반환. 사용자는 --model로 임의 지정 가능.
        return [
            {"alias": "default", "id": self.DEFAULT_MODEL, "note": "Gemini 2.5 Flash 이미지 생성"},
        ]

    def balance(self) -> dict[str, Any]:
        """Gemini는 공식 잔여 쿼터 API가 없음. 등록된 키 수와 각 키의 접근성만 확인.

        키 검증은 lightweight하게 `next(iter(models.list()))`로 단일 페이지만 조회.
        """
        if not self._keys:
            raise RuntimeError("GEMINI_API_KEYS 환경변수를 설정하세요.")

        key_status: list[dict] = []
        all_ok = True
        for key in self._keys:
            hint = key[:8] + "..." + key[-4:]
            try:
                client = self._get_client(key)
                _ = next(iter(client.models.list()))
                key_status.append({"key": hint, "status": "ok"})
            except StopIteration:
                key_status.append({"key": hint, "status": "ok"})
            except Exception as e:
                all_ok = False
                err = str(e)[:120]
                key_status.append({"key": hint, "status": "error", "detail": err})

        return {
            "provider": self.name,
            "available": all_ok,
            "quota_remaining": None,
            "quota_total": None,
            "unit": None,
            "renews_at": None,
            "plan": None,
            "note": "Gemini는 잔여 쿼터 API 미제공. https://ai.dev/rate-limit 에서 확인",
            "details": {
                "num_keys": len(self._keys),
                "keys": key_status,
            },
        }

    def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        width: int | None = None,
        height: int | None = None,
        aspect_ratio: str | None = None,
    ) -> GeneratedImage:
        if not self._keys:
            raise RuntimeError(
                "GEMINI_API_KEYS 환경변수를 설정하세요. (쉼표로 여러 키 구분 가능) "
                "https://aistudio.google.com/apikey 에서 발급 가능합니다."
            )

        last_error: Exception | None = None
        for i, key in enumerate(self._keys):
            client = self._get_client(key)
            try:
                return self._try_generate(client, prompt, output_path, aspect_ratio)
            except Exception as e:
                last_error = e
                if _is_rate_limited(e) and i < len(self._keys) - 1:
                    # 다음 키로 전환 (stderr 진행 메시지는 cli 레벨에서 처리)
                    continue
                raise

        raise last_error  # type: ignore[misc]

    def _try_generate(
        self,
        client: genai.Client,
        prompt: str,
        output_path: Path,
        aspect_ratio: str | None,
    ) -> GeneratedImage:
        response = client.models.generate_content(
            model=self._model,
            contents=f"Generate an image: {prompt}",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        image_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_data = part.inline_data.data
                break

        if image_data is None:
            raise RuntimeError("이미지 생성 실패: 응답에 이미지가 없습니다.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_data)

        return GeneratedImage(
            path=output_path,
            prompt=prompt,
            provider=self.name,
            model=self._model,
        )


def _is_rate_limited(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg
