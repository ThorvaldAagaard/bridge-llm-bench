"""
DeepSeek API client implementation.
"""

import os
from .openai_client import OpenAIClient

openai = None


class DeepSeekClient(OpenAIClient):
    """Client for DeepSeek API (OpenAI-compatible)."""

    MODEL_MAP = {
        "deepseek-R1-0528": "deepseek-reasoner",
        "deepseek-r1-0528": "deepseek-reasoner",
        "deepseek-r1": "deepseek-reasoner",
        "deepseek-V3-0324": "deepseek-chat",
        "deepseek-v3-0324": "deepseek-chat",
        "deepseek-v3": "deepseek-chat",
        "deepseek-v3.2": "deepseek-chat",
        "deepseek-v3.2-exp": "deepseek-chat",
    }

    def __init__(self, model: str, temperature: float = 0.0):
        # Map model name to API model
        api_model = self.MODEL_MAP.get(model, self.MODEL_MAP.get(model.lower(), "deepseek-chat"))

        # Store original name for display
        self._display_name = model

        # Initialize parent with mapped model
        super().__init__(api_model, temperature)

        # Restore display name
        self.model_name = model

        # Get DeepSeek API key
        deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not deepseek_key:
            raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY not set")

        global openai
        if openai is None:
            import openai as _openai
            openai = _openai

        # Reconfigure client with DeepSeek endpoint
        if self.use_v1:
            self.client = openai.OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com/v1",
            )
        else:
            openai.api_key = deepseek_key
            openai.api_base = "https://api.deepseek.com/v1"
