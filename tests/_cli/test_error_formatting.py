"""Tests for CLI exception formatting helpers."""

from openbench._cli.error_formatting import format_exception_with_causes


def test_format_exception_with_nested_cause_chain() -> None:
    """Nested exceptions should include a readable cause chain."""
    try:
        try:
            raise TimeoutError("read timed out")
        except TimeoutError as err:
            raise RuntimeError("Connection error.") from err
    except RuntimeError as err:
        formatted = format_exception_with_causes(err)

    assert "Connection error." in formatted
    assert "Cause chain:" in formatted
    assert "TimeoutError: read timed out" in formatted


def test_format_exception_includes_request_metadata() -> None:
    """Request method and URL should be surfaced when available."""

    class _Request:
        method = "POST"
        url = "https://models.mixlayer.ai/v1/chat/completions"

    class _RequestError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.request = _Request()

    formatted = format_exception_with_causes(_RequestError("Connection error."))

    assert "Request: POST https://models.mixlayer.ai/v1/chat/completions" in formatted


def test_format_exception_falls_back_to_class_name_when_message_empty() -> None:
    """Exceptions without messages should still render useful output."""

    class _NoMessageError(Exception):
        pass

    formatted = format_exception_with_causes(_NoMessageError())

    assert "_NoMessageError" in formatted
