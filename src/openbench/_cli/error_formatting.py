"""Utilities for formatting CLI exceptions with root-cause context."""

from __future__ import annotations


def _message(error: BaseException) -> str:
    """Return a stable human-readable message for an exception."""
    message = str(error).strip()
    return message or error.__class__.__name__


def format_exception_with_causes(error: BaseException, max_depth: int = 8) -> str:
    """Format an exception including request metadata and nested causes."""
    lines = [_message(error)]

    request = getattr(error, "request", None)
    method = getattr(request, "method", None)
    url = getattr(request, "url", None)
    if method and url:
        lines.append(f"Request: {method} {url}")
    elif url:
        lines.append(f"Request URL: {url}")

    chain: list[str] = []
    seen = {id(error)}
    current = error.__cause__ or error.__context__
    while current and len(chain) < max_depth and id(current) not in seen:
        seen.add(id(current))
        chain.append(f"{current.__class__.__name__}: {_message(current)}")
        current = current.__cause__ or current.__context__

    if chain:
        lines.append("Cause chain:")
        lines.extend(f"  - {item}" for item in chain)

    if current and len(chain) >= max_depth:
        lines.append("  - ... (additional nested causes omitted)")

    return "\n".join(lines)
