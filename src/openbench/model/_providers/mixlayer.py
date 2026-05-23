"""Mixlayer provider implementation (OpenAI-compatible)."""

import os
from typing import Any

import httpx
from inspect_ai.model import GenerateConfig
from inspect_ai.model._openai import OpenAIAsyncHttpxClient
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from openai.types.chat import ChatCompletion


class MixlayerAPI(OpenAICompatibleAPI):
    """Mixlayer OpenAI-compatible inference provider."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Remove service prefix if present
        model_name_clean = model_name.replace("mixlayer/", "", 1)
        thinking = model_args.pop("thinking", None)
        self.stream = model_args.pop("stream", True)
        self.stream_options = model_args.pop("stream_options", {"include_usage": True})

        base_url = base_url or os.environ.get(
            "MIXLAYER_BASE_URL", "https://models.mixlayer.ai/v1"
        )
        api_key = api_key or os.environ.get("MIXLAYER_API_KEY")

        if not api_key:
            raise ValueError(
                "Mixlayer API key not found. Set MIXLAYER_API_KEY environment variable."
            )

        timeout_seconds = getattr(config, "timeout", None)
        if timeout_seconds is not None and "http_client" not in model_args:
            model_args["http_client"] = OpenAIAsyncHttpxClient(
                timeout=httpx.Timeout(timeout=timeout_seconds)
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="mixlayer",
            service_base_url="https://models.mixlayer.ai/v1",
            **model_args,
        )

        # Support Mixlayer-specific request-body options via model args.
        self._extra_body = {}
        if thinking is not None:
            self._extra_body["thinking"] = thinking

        if self._extra_body:
            original_create = self.client.chat.completions.create

            def create_with_mixlayer_options(**kwargs):
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                if kwargs["extra_body"] is None:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"].update(self._extra_body)
                return original_create(**kwargs)

            setattr(self.client.chat.completions, "create", create_with_mixlayer_options)

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name

    async def _generate_completion(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ChatCompletion:
        if not self.stream:
            return await super()._generate_completion(request, config)

        stream_request = {
            **request,
            "stream": True,
            "stream_options": self.stream_options,
        }
        stream = await self.client.chat.completions.create(**stream_request)
        return await self._handle_streaming_response(stream)

    async def _handle_streaming_response(self, stream: Any) -> ChatCompletion:
        content = ""
        reasoning_content = ""
        finish_reason = "stop"
        completion_id = ""
        created = 0
        model = self.model_name
        system_fingerprint = None
        usage = None

        async for chunk in stream:
            if getattr(chunk, "id", None):
                completion_id = chunk.id
            if getattr(chunk, "created", None):
                created = chunk.created
            if getattr(chunk, "model", None):
                model = chunk.model
            if getattr(chunk, "system_fingerprint", None):
                system_fingerprint = chunk.system_fingerprint
            if getattr(chunk, "usage", None):
                usage = chunk.usage

            for choice in getattr(chunk, "choices", []) or []:
                delta = choice.delta
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta:
                    reasoning_content += reasoning_delta
                content_delta = getattr(delta, "content", None)
                if content_delta:
                    content += content_delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        completion: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if system_fingerprint is not None:
            completion["system_fingerprint"] = system_fingerprint
        if usage is not None:
            completion["usage"] = usage.model_dump()

        return ChatCompletion.model_validate(completion)
