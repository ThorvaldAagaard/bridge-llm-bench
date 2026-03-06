"""
xAI Grok API client implementation (OpenAI-compatible).
"""

import os
from .openai_client import OpenAIClient

openai = None


class GrokClient(OpenAIClient):
    """Client for xAI Grok API (OpenAI-compatible)."""

    REASONING_PREFIXES = {"grok-4.1-thinking", "grok-4.20-beta1"}

    def __init__(self, model: str, temperature: float = 0.0):
        self._display_name = model

        # Initialize parent with model name
        super().__init__(model, temperature)

        # Restore display name
        self.model_name = model

        # Get xAI API key
        xai_key = os.getenv("XAI_API_KEY")
        if not xai_key:
            raise RuntimeError("XAI_API_KEY environment variable not set")

        global openai
        if openai is None:
            import openai as _openai
            openai = _openai

        # Reconfigure client with xAI endpoint
        if self.use_v1:
            self.client = openai.OpenAI(
                api_key=xai_key,
                base_url="https://api.x.ai/v1",
            )
        else:
            openai.api_key = xai_key
            openai.api_base = "https://api.x.ai/v1"

    @property
    def is_reasoning_model(self) -> bool:
        model_lower = self.model_name.lower()
        return any(model_lower.startswith(p) for p in self.REASONING_PREFIXES)
