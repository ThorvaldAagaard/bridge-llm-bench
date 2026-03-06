"""
Base client class for all LLM API clients.

This module defines the abstract base class that all API clients must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseClient(ABC):
    """
    Abstract base class for all API clients.
    
    Parameters
    ----------
    model_name : str
        Name of the model to use
    temperature : float, optional
        Temperature setting for generation (default: 0.0)
        
    Attributes
    ----------
    model_name : str
        Name of the model
    model : str
        Alias for model_name (for compatibility)
    temperature : float
        Temperature setting
    """
    
    def __init__(self, model_name: str, temperature: float = 0.0):
        """
        Initialize the base client.
        
        Parameters
        ----------
        model_name : str
            Name of the model to use
        temperature : float, optional
            Temperature setting for generation (default: 0.0)
        """
        self.model_name = model_name
        self.model = model_name  # Alias for compatibility
        self.temperature = temperature
    
    @abstractmethod
    def get_completion(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Send a prompt to the model and get a completion.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model
            
        Returns
        -------
        tuple
            A tuple containing:
            - text (str): The generated text response
            - metadata (dict): Dictionary with usage information including:
                - prompt_tokens: Number of tokens in the prompt
                - completion_tokens: Number of tokens in the completion
                
        Raises
        ------
        Exception
            If the API call fails
        """
        raise NotImplementedError("Subclasses must implement get_completion")