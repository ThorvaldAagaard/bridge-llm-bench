"""
OpenAI API client implementation.
"""

import os
from typing import Tuple, Dict, Any

from .base import BaseClient
from ..utils.config import MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS_REASONING, REASONING_MODELS
from ..utils.decorators import exponential_backoff

openai = None


class OpenAIClient(BaseClient):
    """Client for OpenAI API models (GPT, O3, etc.)."""

    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)

        global openai
        if openai is None:
            import openai as _openai
            openai = _openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        self.use_v1 = hasattr(openai, "OpenAI")

        if self.use_v1:
            base_url = os.getenv("OPENAI_API_BASE", None)
            self.client = (
                openai.OpenAI(api_key=api_key, base_url=base_url)
                if base_url
                else openai.OpenAI(api_key=api_key)
            )
        else:
            openai.api_key = api_key
            if os.getenv("OPENAI_API_BASE"):
                openai.api_base = os.getenv("OPENAI_API_BASE")

    @property
    def is_reasoning_model(self) -> bool:
        model_lower = self.model_name.lower()
        return any(model_lower.startswith(prefix) for prefix in REASONING_MODELS)

    @exponential_backoff(max_retries=3, exceptions=(Exception,))
    def get_completion(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Get completion. Same prompt for all models; only API params differ."""
        params = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        if self.is_reasoning_model:
            params["max_completion_tokens"] = MAX_OUTPUT_TOKENS_REASONING
            # Reasoning models may not support temperature
        else:
            params["temperature"] = self.temperature
            params["max_tokens"] = MAX_OUTPUT_TOKENS

        # Retry loop to handle unsupported parameters (temperature, max_tokens)
        for _attempt in range(3):
            try:
                response = self._make_api_call(params)
                return self._parse_response(response)
            except Exception as e:
                error_msg = str(e)
                if self._should_retry_without_parameter(error_msg):
                    params = self._adjust_parameters(params, error_msg)
                    continue
                raise

    def _make_api_call(self, params: Dict[str, Any]) -> Any:
        if self.use_v1:
            return self.client.chat.completions.create(**params)
        else:
            return openai.ChatCompletion.create(**params)

    def _parse_response(self, response: Any) -> Tuple[str, Dict[str, Any]]:
        msg_obj = response.choices[0].message
        if isinstance(msg_obj, dict):
            text = (msg_obj.get("content") or "").strip()
        else:
            text = (getattr(msg_obj, "content", "") or "").strip()

        usage_obj = getattr(response, "usage", {}) or {}
        if isinstance(usage_obj, dict):
            usage = usage_obj
        else:
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
            }

        metadata = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

        return text, metadata

    def _should_retry_without_parameter(self, error_msg: str) -> bool:
        msg = error_msg.lower()
        return (
            ("max_tokens" in msg and "unsupported" in msg)
            or ("temperature" in msg and "unsupported" in msg)
        )

    def _adjust_parameters(self, params: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        if "max_tokens" in error_msg:
            params.pop("max_tokens", None)
            params["max_completion_tokens"] = MAX_OUTPUT_TOKENS
        elif "temperature" in error_msg:
            params.pop("temperature", None)
        return params
