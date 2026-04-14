import os
import time
from pathlib import Path
from typing import Any

import httpx

from mim_cli.providers import GeneratedImage, ImageProvider


# 별명 → (slug, 설명)
MODEL_PRESETS: dict[str, tuple[str, str]] = {
    # Black Forest Labs FLUX 계열
    "flux-schnell": ("black-forest-labs/flux-schnell", "빠르고 저렴 (~$0.003/장)"),
    "flux-dev": ("black-forest-labs/flux-dev", "고품질"),
    "flux-pro": ("black-forest-labs/flux-pro", "프로급"),
    "flux-1.1-pro": ("black-forest-labs/flux-1.1-pro", "최신 프로"),
    "flux-1.1-pro-ultra": ("black-forest-labs/flux-1.1-pro-ultra", "울트라 품질 (4K)"),
    # Stability AI
    "sd-3.5-large": ("stability-ai/stable-diffusion-3.5-large", "SD 3.5 최신"),
    "sd-3.5-large-turbo": ("stability-ai/stable-diffusion-3.5-large-turbo", "SD 3.5 고속"),
    "sd-3.5-medium": ("stability-ai/stable-diffusion-3.5-medium", "SD 3.5 중간 크기"),
    "sd-3": ("stability-ai/stable-diffusion-3", "SD 3"),
    # 빠른 SDXL
    "sdxl-lightning": ("bytedance/sdxl-lightning-4step", "4스텝 고속 SDXL"),
    # Recraft (벡터/일러스트)
    "recraft-v3": ("recraft-ai/recraft-v3", "벡터/일러스트 강점"),
    "recraft-v3-svg": ("recraft-ai/recraft-v3-svg", "SVG 출력"),
    # Ideogram (텍스트 렌더링)
    "ideogram-v2": ("ideogram-ai/ideogram-v2", "텍스트 렌더링 강점"),
    "ideogram-v2-turbo": ("ideogram-ai/ideogram-v2-turbo", "Ideogram 고속"),
    # Google Imagen
    "imagen-3": ("google/imagen-3", "Google Imagen 3"),
    "imagen-3-fast": ("google/imagen-3-fast", "Imagen 3 고속"),
}


class ReplicateProvider(ImageProvider):
    """Replicate 이미지 생성 (HTTP API 직접 호출)."""

    name = "replicate"
    DEFAULT_MODEL_ALIAS = "flux-schnell"
    BASE_URL = "https://api.replicate.com/v1"
    POLL_INTERVAL = 1.0

    def __init__(
        self,
        api_token: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self._token = (
            api_token
            or os.environ.get("REPLICATE_API_TOKEN", "").strip()
            or os.environ.get("REPLICATE_API_KEY", "").strip()
        )
        alias = model or self.DEFAULT_MODEL_ALIAS
        preset = MODEL_PRESETS.get(alias)
        if preset:
            self._model = preset[0]
            self._model_label = alias
        else:
            # 원시 "owner/name" 슬러그 입력
            self._model = alias
            self._model_label = alias
        self._timeout = timeout

    def check_auth(self) -> bool:
        return bool(self._token)

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {"alias": alias, "id": slug, "note": note}
            for alias, (slug, note) in MODEL_PRESETS.items()
        ]

    def balance(self) -> dict[str, Any]:
        """Replicate는 공식 잔여 크레딧 API 미제공. 계정 메타데이터만 반환."""
        if not self._token:
            raise RuntimeError("REPLICATE_API_TOKEN 환경변수를 설정하세요.")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{self.BASE_URL}/account", headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return {
            "provider": self.name,
            "available": True,
            "quota_remaining": None,
            "quota_total": None,
            "unit": None,
            "renews_at": None,
            "plan": data.get("type"),
            "note": "Replicate는 잔여 크레딧 API 미제공. https://replicate.com/account/billing 참조",
            "details": {
                "username": data.get("username"),
                "name": data.get("name"),
                "type": data.get("type"),
                "github_url": data.get("github_url"),
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
        if not self._token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN 환경변수를 설정하세요. "
                "https://replicate.com/account/api-tokens 에서 발급 가능합니다."
            )

        inputs: dict = {
            "prompt": prompt,
            "output_format": "png",
            "num_outputs": 1,
        }
        if aspect_ratio:
            inputs["aspect_ratio"] = aspect_ratio

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }

        with httpx.Client(timeout=self._timeout) as client:
            url = f"{self.BASE_URL}/models/{self._model}/predictions"
            resp = client.post(url, headers=headers, json={"input": inputs})
            resp.raise_for_status()
            pred = resp.json()

            while pred.get("status") in ("starting", "processing"):
                time.sleep(self.POLL_INTERVAL)
                poll_url = pred["urls"]["get"]
                resp = client.get(poll_url, headers=headers)
                resp.raise_for_status()
                pred = resp.json()

            if pred.get("status") != "succeeded":
                err = pred.get("error") or pred.get("status")
                raise RuntimeError(f"이미지 생성 실패: {err}")

            output = pred.get("output")
            if isinstance(output, list):
                if not output:
                    raise RuntimeError("이미지 생성 실패: 응답이 비어있습니다.")
                image_url = output[0]
            elif isinstance(output, str):
                image_url = output
            else:
                raise RuntimeError(f"이미지 생성 실패: 알 수 없는 응답 형식 ({type(output)})")

            img_resp = client.get(image_url)
            img_resp.raise_for_status()
            image_data = img_resp.content

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_data)

        return GeneratedImage(
            path=output_path,
            prompt=prompt,
            provider=self.name,
            model=self._model_label,
        )
