"""
Hand parsing utilities for Bridge hands.

This module provides functions to format and parse Bridge hands
from various representations.
"""

from typing import List, Dict


def _id2card(card_id: int) -> tuple[str, str]:
    """
    Convert card ID to suit and rank.
    
    Parameters
    ----------
    card_id : int
        Card ID (0-51)
        
    Returns
    -------
    tuple
        (suit, rank) where suit is in 'CDHS' and rank is in '23456789TJQKA'
    """
    suits = "CDHS"
    ranks = "23456789TJQKA"
    return suits[card_id // 13], ranks[card_id % 13]


def format_hand(card_ids: List[int]) -> str:
    """
    Format a hand from card IDs to standard Bridge notation.
    
    Parameters
    ----------
    card_ids : list of int
        List of card IDs (0-51)
        
    Returns
    -------
    str
        Formatted hand like "S:AKQ H:JT9 D:876 C:432"
        
    Examples
    --------
    >>> format_hand([51, 50, 49, 38, 37, 36, 25, 24, 23, 12, 11, 10, 9])
    'S:AKQ H:AKQ D:AKQ C:AKQJ'
    """
    by_suit = {"S": [], "H": [], "D": [], "C": []}
    
    for cid in card_ids:
        suit, rank = _id2card(cid)
        by_suit[suit].append(rank)
    
    # Sort ranks in descending order
    order = "AKQJT98765432"
    for suit in by_suit:
        by_suit[suit].sort(key=lambda r: order.index(r))
    
    # Format as "S:xxx H:xxx D:xxx C:xxx"
    parts = []
    for suit in "SHDC":
        if by_suit[suit]:
            parts.append(f"{suit}:{''.join(by_suit[suit])}")
        else:
            parts.append(f"{suit}:-")
    
    return " ".join(parts)


def format_auction(bid_ids: List[int]) -> str:
    """
    Format auction from bid IDs to readable format.
    
    Parameters
    ----------
    bid_ids : list of int
        List of bid IDs
        
    Returns
    -------
    str
        Formatted auction like "1C - Pass - 1H - Pass"
        
    Notes
    -----
    Uses get_bid_from_id from bid_parser module
    """
    from .bid_parser import get_bid_from_id
    
    if not bid_ids:
        return "None"
    
    bids = [get_bid_from_id(bid_id) for bid_id in bid_ids]
    return " - ".join(bids)


def parse_hand_string(hand_str: str) -> Dict[str, List[str]]:
    """
    Parse a hand string into suits and cards.
    
    Parameters
    ----------
    hand_str : str
        Hand in format "S:AKQ H:JT9 D:876 C:432"
        
    Returns
    -------
    dict
        Dictionary with suits as keys and list of cards as values
        
    Examples
    --------
    >>> parse_hand_string("S:AKQ H:JT9 D:876 C:432")
    {'S': ['A', 'K', 'Q'], 'H': ['J', 'T', '9'], 'D': ['8', '7', '6'], 'C': ['4', '3', '2']}
    """
    result = {"S": [], "H": [], "D": [], "C": []}
    
    parts = hand_str.strip().split()
    for part in parts:
        if ":" in part:
            suit, cards = part.split(":")
            if suit in result and cards != "-":
                result[suit] = list(cards)
    
    return result


def count_hcp(hand_str: str) -> int:
    """
    Count High Card Points in a hand.
    
    Parameters
    ----------
    hand_str : str
        Hand in format "S:AKQ H:JT9 D:876 C:432"
        
    Returns
    -------
    int
        Total HCP (A=4, K=3, Q=2, J=1)
        
    Examples
    --------
    >>> count_hcp("S:AKQ H:JT9 D:876 C:432")
    10
    """
    hcp_values = {"A": 4, "K": 3, "Q": 2, "J": 1}
    total = 0
    
    hand = parse_hand_string(hand_str)
    for suit, cards in hand.items():
        for card in cards:
            total += hcp_values.get(card, 0)
    
    return total