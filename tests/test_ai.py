from unittest.mock import patch, MagicMock
from pathlib import Path
from meme_cli.ai import MetadataGenerator, GeneratedMetadata


def test_generated_metadata_structure():
    meta = GeneratedMetadata(
        name="테스트",
        description="설명",
        tags=["충격", "반응"],
        emotions=["충격"],
        context=["반응"],
    )
    assert meta.name == "테스트"
    assert isinstance(meta.tags, list)


@patch("meme_cli.ai.subprocess.run")
def test_generate_calls_claude_with_path(mock_run, tmp_path):
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n")  # 더미 파일

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout='{"name":"충격","description":"놀란 표정","tags":["충격"],"emotions":["충격"],"context":["반응"]}',
    )

    gen = MetadataGenerator()
    meta = gen.generate(img_path)

    # claude --model haiku --print 가 파일 경로를 포함한 프롬프트로 호출됐는지 확인
    call_args = mock_run.call_args
    cmd = call_args[0][0]
    assert cmd[0] == "claude"
    assert "--model" in cmd
    assert "haiku" in cmd[cmd.index("--model") + 1]
    assert "--print" in cmd
    assert str(img_path) in cmd[-1]

    assert meta.name == "충격"
    assert meta.tags == ["충격"]


@patch("meme_cli.ai.subprocess.run")
def test_parse_json_in_markdown_block(mock_run, tmp_path):
    """Claude가 ```json 블록으로 감싸 응답해도 파싱 성공"""
    img_path = tmp_path / "meme.gif"
    img_path.write_bytes(b"GIF89a")

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout='```json\n{"name":"폭발","description":"폭발 장면","tags":["폭발"],"emotions":["극적"],"context":["전환"]}\n```',
    )

    gen = MetadataGenerator()
    meta = gen.generate(img_path)
    assert meta.name == "폭발"
