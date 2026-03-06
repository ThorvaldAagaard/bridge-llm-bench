"""
Baidu ERNIE API client implementation.
"""

import os
import sys
from typing import Tuple, Dict, Any

from .base import BaseClient
from ..utils.config import MAX_OUTPUT_TOKENS
from ..utils.decorators import exponential_backoff


class ErnieClient(BaseClient):
    """Client for Baidu ERNIE API (Qianfan platform)."""

    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)

        try:
            import requests
        except ImportError:
            sys.exit("Please install requests: pip install requests")

        self.api_key = os.getenv("QIANFAN_API_KEY") or os.getenv("ERNIE_API_KEY")
        self.secret_key = os.getenv("QIANFAN_SECRET_KEY") or os.getenv("ERNIE_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "QIANFAN_API_KEY/QIANFAN_SECRET_KEY (or ERNIE_API_KEY/ERNIE_SECRET_KEY) not set"
            )

        self._access_token = None

    def _get_access_token(self) -> str:
        """Get or refresh Baidu access token."""
        if self._access_token:
            return self._access_token

        import requests

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }
        resp = requests.post(url, params=params, timeout=30)
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]
        return self._access_token

    @exponential_backoff(max_retries=3, exceptions=(Exception,))
    def get_completion(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Get completion from ERNIE."""
        import requests

        token = self._get_access_token()

        url = (
            f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/"
            f"{self._endpoint_for_model()}?access_token={token}"
        )

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": max(0.01, self.temperature),  # ERNIE requires > 0
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }

        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "error_code" in data:
            # Token expired
            if data["error_code"] == 111:
                self._access_token = None
                raise RuntimeError(f"ERNIE token expired: {data}")
            raise RuntimeError(f"ERNIE API error: {data}")

        text = data.get("result", "")
        usage = data.get("usage", {})
        metadata = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

        return text.strip(), metadata

    def _endpoint_for_model(self) -> str:
        """Map model name to ERNIE API endpoint slug."""
        model_lower = self.model_name.lower()
        if "5.0" in model_lower or "5" in model_lower:
            return "ernie-4.0-turbo-128k"  # latest available endpoint
        if "4.0" in model_lower:
            return "ernie-4.0-8k"
        return "ernie-4.0-turbo-128k"
