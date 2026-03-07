"""Unit tests for Mixlayer provider request body extensions."""

from unittest.mock import patch

from inspect_ai.model import GenerateConfig

from openbench.model._providers.mixlayer import MixlayerAPI


class _DummyCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


class _DummyChat:
    def __init__(self) -> None:
        self.completions = _DummyCompletions()


class _DummyHttpClient:
    def __init__(self) -> None:
        self.event_hooks = {"request": [], "response": []}


class _DummyAsyncOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = _DummyChat()
        self._client = _DummyHttpClient()

    async def close(self) -> None:
        return None


def _make_provider(**kwargs) -> MixlayerAPI:
    with patch(
        "inspect_ai.model._providers.openai_compatible.AsyncOpenAI", _DummyAsyncOpenAI
    ):
        return MixlayerAPI(
            model_name="mixlayer/test-model",
            api_key="test-key",
            base_url="https://models.mixlayer.ai/v1",
            config=GenerateConfig(),
            http_client=object(),
            **kwargs,
        )


def test_thinking_true_injected_into_extra_body() -> None:
    provider = _make_provider(thinking=True)
    provider.client.chat.completions.create(messages=[], model="test-model")

    call = provider.client.chat.completions.calls[-1]
    assert call["extra_body"]["thinking"] is True


def test_thinking_merges_with_existing_extra_body() -> None:
    provider = _make_provider(thinking=True)
    provider.client.chat.completions.create(
        messages=[],
        model="test-model",
        extra_body={"foo": "bar"},
    )

    call = provider.client.chat.completions.calls[-1]
    assert call["extra_body"]["foo"] == "bar"
    assert call["extra_body"]["thinking"] is True


def test_no_thinking_does_not_inject_extra_body() -> None:
    provider = _make_provider()
    provider.client.chat.completions.create(messages=[], model="test-model")

    call = provider.client.chat.completions.calls[-1]
    assert "extra_body" not in call
