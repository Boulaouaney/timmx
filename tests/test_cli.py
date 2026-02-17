import pytest

from timmx.cli import build_parser


def test_root_help_mentions_export(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "export" in captured.out


def test_export_onnx_arguments_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "export",
            "onnx",
            "resnet18",
            "--output",
            "artifacts/model.onnx",
            "--no-check",
            "--dynamic-batch",
        ]
    )
    assert args.command == "export"
    assert args.format == "onnx"
    assert args.model_name == "resnet18"
    assert args.check is False
    assert args.dynamic_batch is True
