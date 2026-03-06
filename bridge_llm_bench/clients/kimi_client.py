"""
Moonshot Kimi API client implementation (OpenAI-compatible).
"""

import os
from .openai_client import OpenAIClient

openai = None


class KimiClient(OpenAIClient):
    """Client for Moonshot Kimi API (OpenAI-compatible)."""

    REASONING_PREFIXES = {"kimi-k2.5-thinking"}

    def __init__(self, model: str, temperature: float = 0.0):
        self._display_name = model

        # Initialize parent with model name
        super().__init__(model, temperature)

        # Restore display name
        self.model_name = model

        # Get Moonshot API key
        moonshot_key = os.getenv("MOONSHOT_API_KEY")
        if not moonshot_key:
            raise RuntimeError("MOONSHOT_API_KEY environment variable not set")

        global openai
        if openai is None:
            import openai as _openai
            openai = _openai

        # Reconfigure client with Moonshot endpoint
        if self.use_v1:
            self.client = openai.OpenAI(
                api_key=moonshot_key,
                base_url="https://api.moonshot.cn/v1",
            )
        else:
            openai.api_key = moonshot_key
            openai.api_base = "https://api.moonshot.cn/v1"

    @property
    def is_reasoning_model(self) -> bool:
        model_lower = self.model_name.lower()
        return any(model_lower.startswith(p) for p in self.REASONING_PREFIXES)
