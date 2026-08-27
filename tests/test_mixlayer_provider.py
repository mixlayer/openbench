"""Unit tests for Mixlayer provider request body extensions."""

from unittest.mock import patch

from inspect_ai.model import GenerateConfig
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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


def test_top_k_is_included_in_completion_extra_body() -> None:
    provider = _make_provider()

    params = provider.completion_params(GenerateConfig(top_k=20), tools=False)

    assert params["extra_body"]["top_k"] == 20


def test_explicit_extra_body_top_k_takes_precedence() -> None:
    provider = _make_provider()

    params = provider.completion_params(
        GenerateConfig(top_k=20, extra_body={"top_k": 10}), tools=False
    )

    assert params["extra_body"]["top_k"] == 10


class _AsyncStream:
    def __init__(self, chunks) -> None:
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _AsyncCompletions:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return _AsyncStream(
                [
                    ChatCompletionChunk.model_validate(
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": 123,
                            "model": "test-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "reasoning_content": "first ",
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ),
                    ChatCompletionChunk.model_validate(
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": 123,
                            "model": "test-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": "Answer",
                                        "reasoning_content": "second",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ),
                    ChatCompletionChunk.model_validate(
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": 123,
                            "model": "test-model",
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 5,
                                "completion_tokens": 7,
                                "total_tokens": 12,
                            },
                        }
                    ),
                ]
            )
        return ChatCompletion.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 123,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Answer"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


class _AsyncChat:
    def __init__(self) -> None:
        self.completions = _AsyncCompletions()


class _AsyncOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = _AsyncChat()
        self._client = _DummyHttpClient()

    async def close(self) -> None:
        return None


def _make_async_provider(**kwargs) -> MixlayerAPI:
    with patch(
        "inspect_ai.model._providers.openai_compatible.AsyncOpenAI", _AsyncOpenAI
    ):
        return MixlayerAPI(
            model_name="mixlayer/test-model",
            api_key="test-key",
            base_url="https://models.mixlayer.ai/v1",
            config=GenerateConfig(),
            http_client=object(),
            **kwargs,
        )


async def test_streaming_accumulates_reasoning_content() -> None:
    provider = _make_async_provider()
    completion = await provider._generate_completion(
        {"messages": [], "model": "test-model"},
        GenerateConfig(),
    )

    call = provider.client.chat.completions.calls[-1]
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}
    assert completion.choices[0].message.content == "Answer"
    assert getattr(completion.choices[0].message, "reasoning_content") == (
        "first second"
    )
    assert completion.usage.prompt_tokens == 5
    assert completion.usage.completion_tokens == 7


async def test_streaming_can_be_disabled() -> None:
    provider = _make_async_provider(stream=False)
    completion = await provider._generate_completion(
        {"messages": [], "model": "test-model"},
        GenerateConfig(),
    )

    call = provider.client.chat.completions.calls[-1]
    assert "stream" not in call
    assert completion.choices[0].message.content == "Answer"
