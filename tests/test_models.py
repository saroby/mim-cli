from meme_cli.models import MediaItem
import uuid


def test_media_item_defaults():
    item = MediaItem(
        path="/tmp/test.gif",
        media_type="gif",
        name="테스트 밈",
        description="테스트용 GIF",
        tags=["test"],
        emotions=["neutral"],
        context=["testing"],
    )
    assert uuid.UUID(item.id)  # 유효한 UUID
    assert item.created_at != ""
    assert item.updated_at != ""


def test_media_item_roundtrip():
    item = MediaItem(
        path="/tmp/pikachu.png",
        media_type="image",
        name="놀란 피카츄",
        description="충격받은 피카츄 밈",
        tags=["pikachu", "pokemon"],
        emotions=["shocked"],
        context=["reaction", "plot-twist"],
    )
    d = item.to_dict()
    restored = MediaItem.from_dict(d)
    assert restored.id == item.id
    assert restored.tags == ["pikachu", "pokemon"]
    assert restored.emotions == ["shocked"]
