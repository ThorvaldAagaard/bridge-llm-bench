"""
Anthropic API client implementation.
"""

import os
import sys
from typing import Tuple, Dict, Any

from .base import BaseClient
from ..utils.config import MAX_OUTPUT_TOKENS
from ..utils.decorators import exponential_backoff


class AnthropicClient(BaseClient):
    """Client for Anthropic's Claude API."""

    def __init__(self, model_name: str, temperature: float = 0.0):
        super().__init__(model_name, temperature)

        try:
            from anthropic import Anthropic
        except ImportError:
            sys.exit("Please install the anthropic library: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY is not set.")

        self.client = Anthropic(api_key=api_key)

    @exponential_backoff(max_retries=3, exceptions=(Exception,))
    def get_completion(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Get completion from Claude. Uses the same prompt as all other models."""
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        text = response.content[0].text if response.content else ""

        usage = response.usage
        metadata = {
            "prompt_tokens": usage.input_tokens if usage else 0,
            "completion_tokens": usage.output_tokens if usage else 0,
        }

        return text.strip(), metadata
