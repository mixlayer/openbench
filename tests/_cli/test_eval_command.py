"""Simple unit tests for eval command."""

from typer.testing import CliRunner
from openbench._cli import app

runner = CliRunner()


def test_eval_requires_benchmark():
    """Test eval command requires a benchmark argument."""
    result = runner.invoke(app, ["eval"])
    assert result.exit_code != 0


def test_invalid_limit():
    """Test invalid limit parameter."""
    result = runner.invoke(app, ["eval", "mmlu", "--limit", "invalid"])
    assert result.exit_code != 0


def test_invalid_display():
    """Test invalid display parameter."""
    result = runner.invoke(app, ["eval", "mmlu", "--display", "invalid"])
    assert result.exit_code != 0


def test_invalid_sandbox():
    """Test invalid sandbox parameter."""
    result = runner.invoke(app, ["eval", "mmlu", "--sandbox", "invalid"])
    assert result.exit_code != 0


def test_invalid_reasoning_effort():
    """Test invalid reasoning effort parameter."""
    result = runner.invoke(app, ["eval", "mmlu", "--reasoning-effort", "invalid"])
    assert result.exit_code != 0


def test_eval_help_includes_top_k():
    """Test top-k is exposed as an eval command option."""
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "--top-k" in result.output
    assert "BENCH_TOP_K" in result.output


def test_eval_forwards_system_message_from_environment(monkeypatch):
    """Test system-message is exposed and forwarded from its environment variable."""
    captured = {}
    monkeypatch.setattr(
        "openbench._cli.eval_command.load_task", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "openbench._cli.eval_command.eval",
        lambda **kwargs: captured.update(kwargs) or [],
    )
    monkeypatch.setattr(
        "openbench._cli.eval_command.patch_display_results", lambda: None
    )

    result = runner.invoke(
        app,
        ["eval", "mmlu", "--display", "none"],
        env={"BENCH_SYSTEM_MESSAGE": "\n\n"},
    )

    assert result.exit_code == 0
    assert captured["system_message"] == "\n\n"

    captured.clear()
    result = runner.invoke(
        app,
        ["eval", "mmlu", "--display", "none", "--system-message", "\n\n"],
    )

    assert result.exit_code == 0
    assert captured["system_message"] == "\n\n"

    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "--system-message" in result.output


def test_eval_forwards_presence_penalty(monkeypatch):
    """Test presence-penalty is forwarded from the CLI and environment."""
    captured = {}
    monkeypatch.setattr(
        "openbench._cli.eval_command.load_task", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "openbench._cli.eval_command.eval",
        lambda **kwargs: captured.update(kwargs) or [],
    )
    monkeypatch.setattr(
        "openbench._cli.eval_command.patch_display_results", lambda: None
    )

    result = runner.invoke(
        app,
        ["eval", "mmlu", "--display", "none", "--presence-penalty", "0.5"],
    )

    assert result.exit_code == 0
    assert captured["presence_penalty"] == 0.5

    captured.clear()
    result = runner.invoke(
        app,
        ["eval", "mmlu", "--display", "none"],
        env={"BENCH_PRESENCE_PENALTY": "0.75"},
    )

    assert result.exit_code == 0
    assert captured["presence_penalty"] == 0.75
