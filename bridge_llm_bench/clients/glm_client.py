"""
Zhipu GLM API client implementation (OpenAI-compatible).
"""

import os
from .openai_client import OpenAIClient

openai = None


class GLMClient(OpenAIClient):
    """Client for Zhipu GLM API (OpenAI-compatible)."""

    def __init__(self, model: str, temperature: float = 0.0):
        self._display_name = model

        # Initialize parent with model name
        super().__init__(model, temperature)

        # Restore display name
        self.model_name = model

        # Get Zhipu API key
        zhipu_key = os.getenv("ZHIPU_API_KEY")
        if not zhipu_key:
            raise RuntimeError("ZHIPU_API_KEY environment variable not set")

        global openai
        if openai is None:
            import openai as _openai
            openai = _openai

        # Reconfigure client with Zhipu endpoint
        if self.use_v1:
            self.client = openai.OpenAI(
                api_key=zhipu_key,
                base_url="https://open.bigmodel.cn/api/paas/v4/",
            )
        else:
            openai.api_key = zhipu_key
            openai.api_base = "https://open.bigmodel.cn/api/paas/v4/"
