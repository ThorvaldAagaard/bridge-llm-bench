"""
API client implementations for various LLM providers.

This module provides a factory function to create the appropriate client
based on the model name.
"""

from typing import Union
from .base import BaseClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .deepseek_client import DeepSeekClient
from .qwen_client import QwenClient
from .grok_client import GrokClient
from .glm_client import GLMClient
from .ernie_client import ErnieClient
from .kimi_client import KimiClient
from .minimax_client import MiniMaxClient
from .xiaomi_client import XiaomiClient

# Client mapping by model prefix
CLIENT_MAPPING = {
    "o3": OpenAIClient,
    "chatgpt": OpenAIClient,
    "gpt": OpenAIClient,
    "deepseek": DeepSeekClient,
    "claude": AnthropicClient,
    "gemini": GeminiClient,
    "qwen": QwenClient,
    "grok": GrokClient,
    "glm": GLMClient,
    "ernie": ErnieClient,
    "kimi": KimiClient,
    "minimax": MiniMaxClient,
    "mimo": XiaomiClient,
}


def get_client(model_name: str, temperature: float = 0.0) -> BaseClient:
    """
    Factory function to instantiate the correct API client.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "gpt-4o", "claude-3-opus")
    temperature : float, optional
        Temperature setting for the model (default: 0.0)

    Returns
    -------
    BaseClient
        Instance of the appropriate client class

    Raises
    ------
    ValueError
        If model prefix is not supported
    """
    model_lower = model_name.lower()

    for prefix, client_class in CLIENT_MAPPING.items():
        if model_lower.startswith(prefix):
            return client_class(model_name, temperature)

    raise ValueError(
        f"Unsupported model prefix for '{model_name}'. "
        f"Supported prefixes: {list(CLIENT_MAPPING.keys())}"
    )


__all__ = ["get_client", "BaseClient"]
