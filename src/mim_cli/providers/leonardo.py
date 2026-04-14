import os
import time
from pathlib import Path
from typing import Any

import httpx

from mim_cli.providers import GeneratedImage, ImageProvider


# aspect_ratio → (width, height). Leonardo는 픽셀 단위로 입력받음.
_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
}

# 친숙한 별명 → (UUID, 설명). 사용자는 별명 또는 원시 UUID 입력 가능.
MODEL_PRESETS: dict[str, tuple[str, str]] = {
    # Phoenix 계열 (프롬프트 충실도 최상)
    "phoenix": ("6b645e3a-d64f-4341-a6d8-7a3690fbf042", "Leonardo Phoenix 1.0 — 프롬프트 충실도 최상"),
    "phoenix-1.0": ("6b645e3a-d64f-4341-a6d8-7a3690fbf042", "Leonardo Phoenix 1.0"),
    "phoenix-0.9": ("de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3", "Leonardo Phoenix 0.9"),
    # Flux (Leonardo 호스팅)
    "flux-schnell": ("1dd50843-d653-4516-a8e3-f0238ee453ff", "FLUX Schnell (빠름)"),
    "flux-dev": ("b2614463-296c-462a-9586-aafdb8f00e36", "FLUX Dev (고품질)"),
    # Leonardo XL 계열
    "lightning-xl": ("b24e16ff-06e3-43eb-8d33-4416c2d75876", "빠른 SDXL 파이프라인"),
    "anime-xl": ("e71a1c2f-4f80-4800-934f-2c68979d8cc8", "애니메이션/만화 스타일"),
    "vision-xl": ("5c232a9e-9061-4777-980a-ddc8e65647c6", "사진 사실적"),
    "diffusion-xl": ("1e60896f-3c26-4296-8ecc-53e2afecc132", "범용 SDXL 튜닝"),
    "kino-xl": ("aa77f04e-3eec-4034-9c07-d0f619684628", "영화적 컴포지션"),
    "albedo-xl": ("2067ae52-33fd-4a82-bb92-c2c55e7d2786", "고품질 범용"),
    # SD 기본
    "sdxl": ("16e7060a-803e-4df3-97ee-edcfa5dc9cc8", "Stable Diffusion XL 1.0"),
}


class LeonardoProvider(ImageProvider):
    """Leonardo.ai 이미지 생성 (무료 티어: 150 토큰/일)."""

    name = "leonardo"
    DEFAULT_MODEL_ALIAS = "phoenix"
    BASE_URL = "https://cloud.leonardo.ai/api/rest/v1"
    POLL_INTERVAL = 2.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self._key = api_key or os.environ.get("LEONARDO_API_KEY", "").strip()
        alias = model or self.DEFAULT_MODEL_ALIAS
        preset = MODEL_PRESETS.get(alias)
        if preset:
            self._model_id = preset[0]
            self._model_label = alias
        else:
            # 원시 UUID 입력
            self._model_id = alias
            self._model_label = f"custom:{alias[:8]}"
        self._timeout = timeout

    def check_auth(self) -> bool:
        return bool(self._key)

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {"alias": alias, "id": uuid, "note": note}
            for alias, (uuid, note) in MODEL_PRESETS.items()
        ]

    def balance(self) -> dict[str, Any]:
        """Leonardo 계정 잔여 토큰 조회 (/me 엔드포인트)."""
        if not self._key:
            raise RuntimeError("LEONARDO_API_KEY 환경변수를 설정하세요.")
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{self.BASE_URL}/me", headers=headers)
            resp.raise_for_status()
            data = resp.json()

        details = (data.get("user_details") or [{}])[0]
        subscription_tokens = details.get("subscriptionTokens")
        paid_tokens = details.get("paidTokens") or 0
        total = (subscription_tokens or 0) + (paid_tokens or 0)

        return {
            "provider": self.name,
            "available": bool(self._key),
            "quota_remaining": total if total else subscription_tokens,
            "quota_total": None,  # Leonardo는 "사용 한도"를 별도로 주지 않음 (일일 리셋)
            "unit": "tokens",
            "renews_at": details.get("tokenRenewalDate"),
            "plan": details.get("subscriptionType"),
            "note": None,
            "details": {
                "subscription_tokens": subscription_tokens,
                "subscription_gpt_tokens": details.get("subscriptionGptTokens"),
                "subscription_model_tokens": details.get("subscriptionModelTokens"),
                "paid_tokens": paid_tokens,
                "api_credit": details.get("apiCredit"),
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
        if not self._key:
            raise RuntimeError(
                "LEONARDO_API_KEY 환경변수를 설정하세요. "
                "https://app.leonardo.ai/settings 에서 발급 가능합니다."
            )

        if aspect_ratio and aspect_ratio in _ASPECT_RATIOS:
            w, h = _ASPECT_RATIOS[aspect_ratio]
        elif width and height:
            w, h = width, height
        else:
            w, h = 1024, 1024

        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "prompt": prompt,
            "modelId": self._model_id,
            "width": w,
            "height": h,
            "num_images": 1,
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self.BASE_URL}/generations",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            job = data.get("sdGenerationJob") or {}
            generation_id = job.get("generationId")
            if not generation_id:
                raise RuntimeError(f"이미지 생성 실패: 응답에 generationId 없음 ({data})")

            image_url = self._poll_until_complete(client, headers, generation_id)

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

    def _poll_until_complete(
        self,
        client: httpx.Client,
        headers: dict,
        generation_id: str,
    ) -> str:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            time.sleep(self.POLL_INTERVAL)
            resp = client.get(
                f"{self.BASE_URL}/generations/{generation_id}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            gen = data.get("generations_by_pk") or {}
            status = gen.get("status")

            if status == "COMPLETE":
                images = gen.get("generated_images") or []
                if not images:
                    raise RuntimeError("이미지 생성 실패: generated_images 비어있음")
                return images[0]["url"]

            if status == "FAILED":
                raise RuntimeError(f"이미지 생성 실패: {gen}")

        raise RuntimeError(f"이미지 생성 타임아웃 ({self._timeout}초)")
