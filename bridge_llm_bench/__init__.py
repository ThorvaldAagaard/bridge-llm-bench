"""
Bridge LLM Benchmarking System

A comprehensive benchmarking system for evaluating Large Language Models (LLMs) 
on Contract Bridge bidding tasks using SAYC and 2/1 conventions.
"""

__version__ = "2.0.0"

from .parsers.bid_parser import parse_bid_from_response
from .parsers.data_loader import load_dataset
from .metrics.evaluator import evaluate
from .utils.config import ARENA_LEADERBOARD_MODELS, CONVENTIONS

__all__ = [
    "parse_bid_from_response",
    "load_dataset", 
    "evaluate",
    "ARENA_LEADERBOARD_MODELS",
    "CONVENTIONS"
]