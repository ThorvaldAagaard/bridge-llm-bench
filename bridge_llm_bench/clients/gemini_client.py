"""
Google Gemini API client implementation.
"""

import os
from typing import Tuple, Dict, Any, Optional

from .base import BaseClient
from ..utils.config import MAX_OUTPUT_TOKENS
from ..utils.decorators import exponential_backoff, retry_on_empty

genai = None


class GeminiClient(BaseClient):
    """Client for Google Gemini API."""

    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)

        global genai
        if genai is None:
            import google.generativeai as _genai
            genai = _genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model_ref = genai.GenerativeModel(model_name=self.model)

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    @exponential_backoff(max_retries=3, exceptions=(Exception,))
    @retry_on_empty(max_retries=3, fallback_value=("Pass", {"prompt_tokens": 0, "completion_tokens": 0}))
    def get_completion(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Get completion from Gemini. Same prompt and params as all other models."""
        cfg = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            candidate_count=1,
        )

        response = self._try_generation(prompt, cfg)

        try:
            text = response.text
        except (ValueError, AttributeError):
            text = ""

        if not text or text.strip() == "":
            text = "Pass"

        usage = getattr(response, "usage_metadata", None) if response else None
        metadata = {
            "prompt_tokens": usage.prompt_token_count if usage else 0,
            "completion_tokens": usage.candidates_token_count if usage else 0,
        }

        return text.strip(), metadata

    def _try_generation(self, prompt: str, cfg: Any) -> Optional[Any]:
        """Try to generate content with retry on blocked responses."""
        for attempt in range(3):
            try:
                response = self.model_ref.generate_content(
                    prompt,
                    generation_config=cfg,
                    safety_settings=self.safety_settings,
                )

                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        finish_reason = candidate.finish_reason
                        reason_name = (
                            finish_reason.name
                            if hasattr(finish_reason, "name")
                            else str(finish_reason)
                        )
                        if reason_name in ("SAFETY", "OTHER") or str(finish_reason) == "3":
                            print(f"Gemini response blocked: {reason_name} (attempt {attempt + 1})")
                            continue

                return response

            except Exception as e:
                print(f"[Gemini ERROR] {str(e)} (attempt {attempt + 1})")
                if attempt < 2:
                    continue
                return None

        return None
