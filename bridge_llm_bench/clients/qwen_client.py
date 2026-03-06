"""
Alibaba Qwen API client implementation.

This module provides the client for Alibaba's Qwen models via DashScope API.
"""

import os
import sys
from typing import Tuple, Dict, Any

from .base import BaseClient
from ..utils.decorators import exponential_backoff


class QwenClient(BaseClient):
    """
    Client for Alibaba's Qwen API (DashScope).
    
    Parameters
    ----------
    model_name : str
        Model name (e.g., "qwen3-235b-a22b", "qwen3-235b-a22b-no-thinking")
    temperature : float, optional
        Temperature for generation (default: 0.0)
        
    Raises
    ------
    ValueError
        If QWEN_API_KEY or DASHSCOPE_API_KEY is not set
    ImportError
        If requests library is not installed
        
    Attributes
    ----------
    api_url : str
        DashScope API endpoint URL
    """
    
    def __init__(self, model_name: str, temperature: float = 0.0):
        """
        Initialize Qwen client with API configuration.
        
        Parameters
        ----------
        model_name : str
            Model name
        temperature : float, optional
            Temperature setting (default: 0.0)
        """
        super().__init__(model_name, temperature)
        
        try:
            import requests
        except ImportError:
            sys.exit("Please install the requests library: pip install requests")
        
        # Get API key (support both environment variables)
        self.api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("Environment variable QWEN_API_KEY or DASHSCOPE_API_KEY is not set.")
        
        # DashScope API endpoint
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    @exponential_backoff(max_retries=3, exceptions=(Exception,))
    def get_completion(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get completion from Qwen model via DashScope API.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model
            
        Returns
        -------
        tuple
            (response_text, metadata) where metadata contains token usage
            
        Notes
        -----
        Uses DashScope's text generation API with the following payload structure:
        - model: Model identifier
        - input.prompt: The input prompt
        - parameters.temperature: Temperature setting
        """
        import requests
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        payload = {
            "model": self.model_name,
            "input": {"prompt": prompt},
            "parameters": {"temperature": self.temperature}
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        # Raise exception for HTTP errors
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Extract text from response
        text = data.get("output", {}).get("text", "")
        
        # Extract usage metadata
        usage = data.get("usage", {})
        metadata = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }
        
        return text.strip(), metadata