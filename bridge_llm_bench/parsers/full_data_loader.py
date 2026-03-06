"""
Full data loader that extracts all 4 hands from the dataset.

This module loads complete Bridge deals including all 4 hands,
the full auction sequence, and the correct next bid.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os

from .bid_parser import get_bid_from_id


def load_full_dataset(
    filepath: Path,
    n_records: Optional[int] = None
) -> List[Dict]:
    """
    Load complete Bridge deals from the dataset.
    
    Parameters
    ----------
    filepath : Path
        Path to the dataset file
    n_records : int, optional
        Number of records to load (None for all)
        
    Returns
    -------
    list of dict
        Each dict contains:
        - 'hands': dict with keys 'N', 'E', 'S', 'W' containing card lists
        - 'dealer': int (0=N, 1=E, 2=S, 3=W)
        - 'vulnerable': str
        - 'auction': list of bid strings
        - 'next_bid': str (the correct next bid)
        - 'card_strings': dict with formatted hand strings
    """
    records = []
    
    with open(filepath, 'r') as f:
        while True:
            # Read the line with all data
            line = f.readline()
            if not line:
                break
                
            # Parse the numbers
            numbers = list(map(int, line.strip().split()))
            if len(numbers) < 53:  # Need at least 52 cards + 1 for format
                continue
            
            # Extract the 52 cards (13 for each player)
            cards = numbers[:52]
            
            # Distribute cards to players
            # Cards are in order: all 13 for player 1, then all 13 for player 2, etc.
            hands = {
                'N': cards[0:13],
                'E': cards[13:26], 
                'S': cards[26:39],
                'W': cards[39:52]
            }
            
            # The rest contains auction bids
            # In the format, after the 52 cards, we have:
            # - A sequence of bid IDs (52+ are bids)
            # - The sequence typically starts with valid bids and may contain other data
            
            remaining = numbers[52:]
            if not remaining:
                continue
            
            # Find where the auction ends
            # Valid bid IDs are 52-92 (Pass=52, X=91, XX=92, suits/levels in between)
            auction_ids = []
            answer_id = None
            
            # The auction is the sequence of valid bid IDs
            # We need to find the actual auction pattern
            for i, num in enumerate(remaining):
                if 52 <= num <= 92:  # Valid bid range
                    auction_ids.append(num)
                else:
                    # First non-bid number might be part of a different encoding
                    # Check if this follows a pattern for answer
                    if i > 0 and len(auction_ids) > 0:
                        # The last valid bid in the auction is the answer
                        answer_id = auction_ids[-1]
                        auction_ids = auction_ids[:-1]
                    break
            
            # If no answer found, skip
            if answer_id is None:
                continue
            
            # Convert bid IDs to bid strings
            auction = [get_bid_from_id(bid_id) for bid_id in auction_ids]
            next_bid = get_bid_from_id(answer_id)
            
            # Format hands as strings
            card_strings = {
                pos: format_hand(hand) for pos, hand in hands.items()
            }
            
            # Determine dealer from auction length
            # In Bridge, dealer rotates: N, E, S, W
            # Board 1: N deals, Board 2: E deals, etc.
            dealer = 0  # Assume North for now
            
            record = {
                'hands': hands,
                'dealer': dealer,
                'vulnerable': 'None',  # Not in this dataset
                'auction': auction,
                'next_bid': next_bid,
                'card_strings': card_strings,
                'auction_length': len(auction)
            }
            
            records.append(record)
            
            if n_records and len(records) >= n_records:
                break
    
    return records


def format_hand(card_ids: List[int]) -> str:
    """
    Format a hand from card IDs to standard Bridge notation.
    
    Parameters
    ----------
    card_ids : list of int
        List of 13 card IDs (0-51)
        
    Returns
    -------
    str
        Formatted hand like "S:AKQ H:JT9 D:876 C:432"
    """
    def id_to_card(card_id: int) -> Tuple[str, str]:
        """Convert card ID to suit and rank."""
        suits = "CDHS"
        ranks = "23456789TJQKA"
        return suits[card_id // 13], ranks[card_id % 13]
    
    by_suit = {"S": [], "H": [], "D": [], "C": []}
    
    for cid in sorted(card_ids):
        suit, rank = id_to_card(cid)
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


def get_next_player(auction_length: int, dealer: int = 0) -> str:
    """
    Determine which player should bid next.
    
    Parameters
    ----------
    auction_length : int
        Number of bids so far
    dealer : int
        Dealer position (0=N, 1=E, 2=S, 3=W)
        
    Returns
    -------
    str
        Position of next player ('N', 'E', 'S', 'W')
    """
    positions = ['N', 'E', 'S', 'W']
    next_pos = (dealer + auction_length) % 4
    return positions[next_pos]


def count_hcp(card_ids: List[int]) -> int:
    """
    Count High Card Points in a hand.
    
    Parameters
    ----------
    card_ids : list of int
        List of card IDs
        
    Returns
    -------
    int
        Total HCP (A=4, K=3, Q=2, J=1)
    """
    hcp = 0
    for card_id in card_ids:
        rank = card_id % 13
        if rank == 12:  # Ace
            hcp += 4
        elif rank == 11:  # King
            hcp += 3
        elif rank == 10:  # Queen
            hcp += 2
        elif rank == 9:   # Jack
            hcp += 1
    return hcp