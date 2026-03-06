"""
Common decorators for retry logic and backoff strategies.

This module provides reusable decorators for implementing
robust retry and exponential backoff patterns.
"""

import functools
import time
import random
from typing import Callable, Any, Type, Tuple, Optional


def exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True
) -> Callable:
    """
    Decorator for exponential backoff retry logic.
    
    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retry attempts (default: 3)
    initial_delay : float, optional  
        Initial delay in seconds between retries (default: 1.0)
    max_delay : float, optional
        Maximum delay in seconds between retries (default: 60.0)
    exponential_base : float, optional
        Base for exponential backoff calculation (default: 2.0)
    exceptions : tuple, optional
        Tuple of exception types to retry on (default: (Exception,))
    jitter : bool, optional
        Whether to add random jitter to delays (default: True)
        
    Returns
    -------
    callable
        Decorated function with retry logic
        
    Examples
    --------
    >>> @exponential_backoff(max_retries=3, initial_delay=1.0)
    ... def api_call():
    ...     # Makes an API call that might fail
    ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        raise
                        
                    # Calculate next delay with exponential backoff
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())
                    else:
                        actual_delay = delay
                        
                    print(f"Retry {attempt + 1}/{max_retries} after {actual_delay:.1f}s delay. Error: {str(e)}")
                    time.sleep(actual_delay)
                    
                    # Exponential backoff
                    delay = min(delay * exponential_base, max_delay)
                    
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_on_empty(
    max_retries: int = 3,
    fallback_value: Optional[Any] = None,
    delay: float = 0.5
) -> Callable:
    """
    Decorator to retry when function returns empty/None values.
    
    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retry attempts (default: 3)
    fallback_value : Any, optional
        Value to return if all retries fail (default: None)
    delay : float, optional
        Delay in seconds between retries (default: 0.5)
        
    Returns
    -------
    callable
        Decorated function with retry logic for empty responses
        
    Examples
    --------
    >>> @retry_on_empty(max_retries=2, fallback_value="Pass")
    ... def get_llm_response():
    ...     # Gets response that might be empty
    ...     return ""
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries + 1):
                result = func(*args, **kwargs)
                
                # Check if result is non-empty
                # For strings, check if not empty after stripping
                if isinstance(result, str):
                    if result.strip():
                        return result
                # For non-strings, None is empty but 0, False, [] etc are valid
                elif result is not None:
                    return result
                    
                if attempt < max_retries:
                    print(f"Empty response, retrying {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                    
            # All retries failed, return fallback
            return fallback_value
            
        return wrapper
    return decorator


def rate_limit(
    calls_per_second: float = 10.0
) -> Callable:
    """
    Decorator to enforce rate limiting on function calls.
    
    Parameters
    ----------
    calls_per_second : float, optional
        Maximum number of calls per second (default: 10.0)
        
    Returns
    -------
    callable
        Decorated function with rate limiting
        
    Examples
    --------
    >>> @rate_limit(calls_per_second=5.0)
    ... def api_call():
    ...     # Makes rate-limited API call
    ...     pass
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
            
        return wrapper
    return decorator