"""
Bid parsing module for extracting and normalizing Bridge bids from text.

This module provides functions to parse Bridge bids from various text formats,
including handling malformed responses, truncations, and special cases.
"""

import re
from typing import Optional, Dict, List

# Bid ID to string mapping
_SUITS = ["C", "D", "H", "S", "NT"]
BID_ID2STR = {52: "Pass", 53: "X", 54: "XX"}
_id = 55
for lvl in range(1, 8):
    for s in _SUITS:
        BID_ID2STR[_id] = f"{lvl}{s}"
        _id += 1

# Regular expressions for bid parsing
BID_REGEX = re.compile(r"\b(PASS|P|X|XX|[1-7](?:C|D|H|S|NT))\b", re.IGNORECASE)
FINAL_BID_REGEX = re.compile(r"FINAL BID:\s*([A-Z0-9]+)", re.IGNORECASE)
STRUCTURED_BID_REGEX = re.compile(r"(?:MY BID IS|I BID|CALL|BID):\s*([A-Z0-9]+)", re.IGNORECASE)
NUMBER_REGEX = re.compile(r"\b([1-7])\b")
SUIT_REGEX = re.compile(r"\b(clubs?|diamonds?|hearts?|spades?|notrump|nt)\b", re.IGNORECASE)


def get_bid_from_id(bid_id: int) -> str:
    """
    Convert numeric bid ID to string representation.
    
    Parameters
    ----------
    bid_id : int
        Numeric bid ID to convert
        
    Returns
    -------
    str
        String representation of the bid (e.g., "1NT", "Pass", "?42")
        
    Notes
    -----
    Common bid IDs:
    - 52: Pass
    - 53: X (Double)
    - 54: XX (Redouble)
    - 55-90: Level/suit combinations (1C through 7NT)
    
    Examples
    --------
    >>> get_bid_from_id(52)
    'Pass'
    >>> get_bid_from_id(55)
    '1C'
    >>> get_bid_from_id(999)
    '?999'
    """
    # Direct lookup
    if bid_id in BID_ID2STR:
        return BID_ID2STR[bid_id]
    
    # Common alternative mappings
    if bid_id == 0:
        return "Pass"
    elif bid_id == 1:
        return "1C"
    
    # Try offset corrections for problematic IDs
    for offset in [-1, 1, -52, -53, -54]:
        adjusted_id = bid_id + offset
        if adjusted_id in BID_ID2STR:
            return BID_ID2STR[adjusted_id]
    
    # Unknown ID
    return f"?{bid_id}"


def parse_bid_from_response(text: str) -> str:
    """
    Extract the first valid bid from an LLM's response text.
    
    Parameters
    ----------
    text : str
        The raw response text from the LLM
        
    Returns
    -------
    str
        The extracted bid in normalized format (e.g., "1NT", "PASS", "X")
        Returns "?" if no valid bid found
        
    Notes
    -----
    This function handles various response formats:
    - Standard: "1NT", "2C", "Pass", "X", "XX"
    - O3-style: "FINAL BID: 2H"
    - Claude-style: "MY BID IS: 1NT"
    - Malformed: "?42" → "2C", "1N" → "1NT"
    - Descriptive: "I recommend passing" → "PASS"
    
    Examples
    --------
    >>> parse_bid_from_response("1NT")
    '1NT'
    >>> parse_bid_from_response("FINAL BID: 2H")
    '2H'
    >>> parse_bid_from_response("I think we should pass")
    'PASS'
    >>> parse_bid_from_response("?42")
    '2C'
    """
    if not text:
        return "?"

    # Strip <think>...</think> blocks from reasoning models (DeepSeek R1, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if not text:
        return "?"

    text_upper = text.upper().strip()
    
    # Check for O3-style FINAL BID format first
    final_bid_match = FINAL_BID_REGEX.search(text_upper)
    if final_bid_match:
        potential_bid = final_bid_match.group(1)
        if _is_valid_bid(potential_bid):
            return _normalize_bid(potential_bid)
    
    # Check for structured formats (Claude style)
    structured_match = STRUCTURED_BID_REGEX.search(text_upper)
    if structured_match:
        potential_bid = structured_match.group(1)
        if _is_valid_bid(potential_bid):
            return _normalize_bid(potential_bid)
    
    # Standard regex for bid tokens
    match = BID_REGEX.search(text_upper)
    if match:
        return _normalize_bid(match.group(1))
    
    # Handle special cases like "?42"
    if text.startswith("?") and len(text) > 1:
        # First try as bid ID
        if text[1:].isdigit():
            bid_id = int(text[1:])
            bid = get_bid_from_id(bid_id)
            if not bid.startswith("?"):
                return bid
        
        # If not a valid bid ID, look for the last single digit that's a valid level
        # This handles "?42" -> "2C" (takes the 2, not the 4)
        digits = re.findall(r'\d', text[1:])
        for digit in reversed(digits):
            if digit in "1234567":
                return f"{digit}C"
    
    # Try to extract from partial/malformed responses
    return _extract_bid_from_partial_response(text, text_upper)


def _is_valid_bid(bid: str) -> bool:
    """
    Check if a string is a valid Bridge bid.
    
    Parameters
    ----------
    bid : str
        The bid string to validate
        
    Returns
    -------
    bool
        True if valid Bridge bid, False otherwise
    """
    bid = bid.upper().strip()
    return bool(re.match(r"^(PASS|X|XX|[1-7](?:C|D|H|S|NT))$", bid))


def _normalize_bid(bid: str) -> str:
    """
    Normalize a bid string to standard format.
    
    Parameters
    ----------
    bid : str
        The bid string to normalize
        
    Returns
    -------
    str
        Normalized bid string
    """
    bid = bid.upper().strip()
    
    # Handle common variations
    if bid in ["P", "PASS"]:
        return "Pass"
    elif bid in ["D", "DBL", "DOUBLE"]:
        return "X"
    elif bid in ["R", "RDBL", "REDOUBLE"]:
        return "XX"
    elif bid.endswith("N") and not bid.endswith("NT"):
        # Convert "1N" to "1NT"
        return bid + "T"
    
    return bid


def _extract_bid_from_partial_response(text: str, text_upper: str) -> str:
    """
    Extract bid from partial or descriptive responses.
    
    Parameters
    ----------
    text : str
        Original response text
    text_upper : str
        Uppercase version of response text
        
    Returns
    -------
    str
        Extracted bid or "?" if none found
        
    Notes
    -----
    Handles cases like:
    - "I recommend passing" → "PASS"
    - "The bid is 2" → "2C"
    - "1C with support" → "1C"
    - "Spades at the 1 level" → "1S"
    """
    # First check if there's a standard bid in the text
    match = BID_REGEX.search(text_upper)
    if match:
        return _normalize_bid(match.group(1))
    
    # Check for pass variations
    if any(word in text_upper for word in ["PASS", "PASSING", "NO BID"]):
        return "Pass"
    
    # Check for double/redouble
    if "DOUBLE" in text_upper and "REDOUBLE" not in text_upper:
        return "X"
    elif "REDOUBLE" in text_upper:
        return "XX"
    
    # Look for level and suit mentions
    level_match = NUMBER_REGEX.search(text)
    suit_match = SUIT_REGEX.search(text)
    
    if level_match:
        level = level_match.group(1)
        
        # Map suit words to symbols
        suit_map = {
            "CLUB": "C", "CLUBS": "C",
            "DIAMOND": "D", "DIAMONDS": "D",
            "HEART": "H", "HEARTS": "H",
            "SPADE": "S", "SPADES": "S",
            "NOTRUMP": "NT", "NT": "NT"
        }
        
        if suit_match:
            suit_word = suit_match.group(1).upper()
            for key, value in suit_map.items():
                if key in suit_word:
                    return f"{level}{value}"
        
        # Default to clubs if only level mentioned
        return f"{level}C"
    
    # Look for truncated bids like "1N"
    truncated_match = re.search(r"\b([1-7])N\b", text_upper)
    if truncated_match:
        return f"{truncated_match.group(1)}NT"
    
    return "?"