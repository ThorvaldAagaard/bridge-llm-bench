"""
Bytedance Dola (Seed) API client implementation (OpenAI-compatible via Volcengine).
"""

import os
from .openai_client import OpenAIClient

openai = None


class DolaClient(OpenAIClient):
    """Client for Bytedance Dola/Seed API (OpenAI-compatible via Volcengine ARK)."""

    def __init__(self, model: str, temperature: float = 0.0):
        self._display_name = model

        # Initialize parent with model name
        super().__init__(model, temperature)

        # Restore display name
        self.model_name = model

        # Get Volcengine API key
        volc_key = os.getenv("VOLCENGINE_API_KEY") or os.getenv("ARK_API_KEY")
        if not volc_key:
            raise RuntimeError(
                "VOLCENGINE_API_KEY or ARK_API_KEY environment variable not set"
            )

        global openai
        if openai is None:
            import openai as _openai
            openai = _openai

        # Reconfigure client with Volcengine ARK endpoint
        if self.use_v1:
            self.client = openai.OpenAI(
                api_key=volc_key,
                base_url="https://ark.cn-beijing.volces.com/api/v3",
            )
        else:
            openai.api_key = volc_key
            openai.api_base = "https://ark.cn-beijing.volces.com/api/v3"
