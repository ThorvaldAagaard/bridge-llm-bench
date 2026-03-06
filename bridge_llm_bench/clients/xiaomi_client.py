"""
Xiaomi MiMo API client implementation (OpenAI-compatible).

MiMo-V2-Flash is open-source, so the API can be hosted by various providers.
Set XIAOMI_API_BASE to point to your preferred endpoint.
"""

import os
from .openai_client import OpenAIClient

openai = None

# Default API base — Xiaomi's official endpoint
DEFAULT_API_BASE = "https://api.mimo.xiaomi.com/v1"


class XiaomiClient(OpenAIClient):
    """Client for Xiaomi MiMo API (OpenAI-compatible)."""

    REASONING_PREFIXES = {"mimo-v2-flash"}

    def __init__(self, model: str, temperature: float = 0.0):
        self._display_name = model

        # Initialize parent with model name
        super().__init__(model, temperature)

        # Restore display name
        self.model_name = model

        # Get Xiaomi API key
        xiaomi_key = os.getenv("XIAOMI_API_KEY")
        if not xiaomi_key:
            raise RuntimeError("XIAOMI_API_KEY environment variable not set")

        api_base = os.getenv("XIAOMI_API_BASE", DEFAULT_API_BASE)

        global openai
        if openai is None:
            import openai as _openai
            openai = _openai

        # Reconfigure client with Xiaomi/provider endpoint
        if self.use_v1:
            self.client = openai.OpenAI(
                api_key=xiaomi_key,
                base_url=api_base,
            )
        else:
            openai.api_key = xiaomi_key
            openai.api_base = api_base

    @property
    def is_reasoning_model(self) -> bool:
        model_lower = self.model_name.lower()
        return any(model_lower.startswith(p) for p in self.REASONING_PREFIXES)
