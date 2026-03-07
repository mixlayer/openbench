"""Mixlayer provider implementation (OpenAI-compatible)."""

import os
from typing import Any

from inspect_ai.model import GenerateConfig
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI


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

        base_url = base_url or os.environ.get(
            "MIXLAYER_BASE_URL", "https://models.mixlayer.ai/v1"
        )
        api_key = api_key or os.environ.get("MIXLAYER_API_KEY")

        if not api_key:
            raise ValueError(
                "Mixlayer API key not found. Set MIXLAYER_API_KEY environment variable."
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
