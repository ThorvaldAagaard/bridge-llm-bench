"""
MiniMax API client implementation (OpenAI-compatible).
"""

import os
from .openai_client import OpenAIClient

openai = None


class MiniMaxClient(OpenAIClient):
    """Client for MiniMax API (OpenAI-compatible)."""

    def __init__(self, model: str, temperature: float = 0.0):
        self._display_name = model

        # Initialize parent with model name
        super().__init__(model, temperature)

        # Restore display name
        self.model_name = model

        # Get MiniMax API key
        minimax_key = os.getenv("MINIMAX_API_KEY")
        if not minimax_key:
            raise RuntimeError("MINIMAX_API_KEY environment variable not set")

        global openai
        if openai is None:
            import openai as _openai
            openai = _openai

        # Reconfigure client with MiniMax endpoint
        if self.use_v1:
            self.client = openai.OpenAI(
                api_key=minimax_key,
                base_url="https://api.minimax.io/v1",
            )
        else:
            openai.api_key = minimax_key
            openai.api_base = "https://api.minimax.io/v1"
