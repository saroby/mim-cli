"""pHash 기반 시각 중복 판별 유틸리티."""

from __future__ import annotations

from pathlib import Path

from mim_cli.models import MediaItem

# pHash는 64비트 DCT 기반 해시라 밝기/리사이즈/압축 차이에 비교적 강하다.
# 8은 stock photo와 meme 변형에서 작은 보정은 잡되, 다른 이미지 오탐은 줄이는 보수적 기본값이다.
DEFAULT_PHASH_THRESHOLD = 8


def compute_perceptual_hash(path: Path) -> str:
    """이미지/GIF의 첫 프레임을 pHash hex 문자열로 변환한다.

    Pillow가 열 수 없는 비디오 포맷은 호출자가 예외를 잡고 해당 항목만 스킵한다.
    """
    import imagehash
    from PIL import Image, ImageSequence

    with Image.open(path) as image:
        first_frame = next(ImageSequence.Iterator(image))
        frame = first_frame.convert("RGB")
        return str(imagehash.phash(frame))


def perceptual_distance(left: str, right: str) -> int:
    """두 pHash hex 문자열의 Hamming distance."""
    import imagehash

    return imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right)


def find_visual_duplicate(
    items: list[MediaItem],
    perceptual_hash: str,
    threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> MediaItem | None:
    """기존 항목 중 threshold 이내의 첫 시각 중복 항목을 찾는다."""
    for item in sorted(
        items,
        key=lambda candidate: (candidate.created_at, candidate.id),
    ):
        if not item.perceptual_hash:
            continue
        try:
            distance = perceptual_distance(perceptual_hash, item.perceptual_hash)
        except Exception:
            continue
        if distance <= threshold:
            return item
    return None


def find_visual_duplicate_groups(
    items: list[MediaItem],
    threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> list[dict[str, MediaItem | list[MediaItem]]]:
    """pHash가 있는 항목들을 연결 요소로 묶고, 각 그룹의 가장 오래된 항목을 keep으로 둔다."""
    candidates = [item for item in items if item.perceptual_hash]
    if len(candidates) < 2:
        return []

    parent = {item.id: item.id for item in candidates}
    by_id = {item.id: item for item in candidates}

    def find(item_id: str) -> str:
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            try:
                distance = perceptual_distance(
                    left.perceptual_hash,
                    right.perceptual_hash,
                )
            except Exception:
                continue
            if distance <= threshold:
                union(left.id, right.id)

    components: dict[str, list[MediaItem]] = {}
    for item in candidates:
        components.setdefault(find(item.id), []).append(by_id[item.id])

    groups: list[dict[str, MediaItem | list[MediaItem]]] = []
    for component in components.values():
        if len(component) < 2:
            continue
        ordered = sorted(component, key=lambda item: (item.created_at, item.id))
        groups.append({"keep": ordered[0], "duplicates": ordered[1:]})

    groups.sort(key=lambda group: group["keep"].created_at)  # type: ignore[union-attr]
    return groups
