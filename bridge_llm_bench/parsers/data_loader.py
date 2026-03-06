"""
Data loading module for Bridge datasets.

This module handles loading Bridge datasets in both textual and numeric formats,
including automatic format detection and conversion.
"""

import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict

from .bid_parser import get_bid_from_id


# OpenSpiel player ordering
PLAYER_NAMES = {0: "North", 1: "East", 2: "South", 3: "West"}


def ensure_default_dataset(path: Union[Path, str] = "data/open_spiel/test.txt") -> str:
    """
    Ensure the default dataset exists, downloading if necessary.

    Downloads from Google's OpenSpiel storage if file doesn't exist.
    """
    path = Path(path)
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/openspiel-data/bridge/test.txt"
    print(f"Downloading dataset to {path}...")

    urllib.request.urlretrieve(url, path)
    return str(path)


def ensure_train_dataset(path: Union[Path, str] = "data/open_spiel/train.txt") -> str:
    """
    Download the large OpenSpiel training set (1M games, ~400MB).

    Downloads from Google's OpenSpiel storage if file doesn't exist.
    """
    path = Path(path)
    if path.exists():
        print(f"Train dataset already exists at {path}")
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/openspiel-data/bridge/train.txt"
    print(f"Downloading training dataset (~400MB) to {path}...")

    urllib.request.urlretrieve(url, path)
    print(f"Download complete: {path}")
    return str(path)


def load_dataset(
    path: Union[Path, str],
    n_records: Optional[int] = None,
    mode: str = "all_bids",
) -> List[Tuple[str, str, str]]:
    """
    Load Bridge dataset from file, auto-detecting format.

    Parameters
    ----------
    path : Path or str
        Path to the dataset file
    n_records : int, optional
        Maximum number of records to load (default: None for all)
    mode : str
        Test case extraction mode:
        - "all_bids": one test case per bid in the auction (~10x records)
        - "first_non_pass": one test case per game (first opening bid)
        - "last_non_pass": one test case per game (last substantive bid)

    Returns
    -------
    list of tuple
        List of (hand, auction, bid) tuples
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        first_non_empty = next((line for line in fh if line.strip()), "")

    if "|" in first_non_empty:
        return _load_text_format(path, n_records)
    else:
        return _load_numeric_format(path, n_records, mode)


# ── Text format loader (pipe-delimited) ─────────────────────────────

def _load_text_format(path: str, n: Optional[int]) -> List[Tuple[str, str, str]]:
    """Load Bridge data from text format: hand | auction | bid."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "|" not in line:
                continue

            parts = [x.strip() for x in line.split("|")]
            if len(parts) != 3:
                continue

            hand, auction, bid = parts
            records.append((hand, auction, bid.upper()))

            if n and len(records) >= n:
                break

    return records


# ── Numeric format helpers ───────────────────────────────────────────

def _split_game_line(numbers: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """
    Split a game line into deal, auction, and play sections.

    The OpenSpiel numeric format is:
      [52 deal actions (card IDs 0-51)]
      [variable-length auction (bid IDs >= 52)]
      [52 play actions (card IDs 0-51)]
    """
    deal = numbers[:52]
    rest = numbers[52:]

    # Scan for contiguous bid IDs (>= 52). The auction ends when we
    # encounter a card ID (< 52) after at least one bid.
    auction_end = 0
    for i, val in enumerate(rest):
        if val < 52:
            auction_end = i
            break
    else:
        # All remaining values are bids (no play data)
        auction_end = len(rest)

    auction = rest[:auction_end]
    play = rest[auction_end:]
    return deal, auction, play


def _parse_deal_interleaved(deal_actions: List[int]) -> Dict[int, List[int]]:
    """
    Parse 52 deal actions into 4 hands using round-robin distribution.

    OpenSpiel deals cards round-robin: position i goes to player i % 4.
    Players: 0=North, 1=East, 2=South, 3=West.
    """
    hands = {0: [], 1: [], 2: [], 3: []}
    for i, card_id in enumerate(deal_actions):
        player = i % 4
        hands[player].append(card_id)
    return hands


def _find_declarer(auction: List[int], dealer: int) -> Optional[int]:
    """
    Find the declarer given the auction and dealer.

    Returns the player index (0-3), or None if all passed.
    """
    # Find the last suit/NT bid (not Pass/X/XX)
    last_bid_idx = None
    last_bid_id = None
    for i, bid_id in enumerate(auction):
        if 55 <= bid_id <= 89:
            last_bid_idx = i
            last_bid_id = bid_id

    if last_bid_id is None:
        return None  # All passed

    # Denomination of the final contract
    denomination = (last_bid_id - 55) % 5  # 0=C, 1=D, 2=H, 3=S, 4=NT

    # The last bidder and their partner
    last_bidder = (dealer + last_bid_idx) % 4
    partner = (last_bidder + 2) % 4

    # Declarer = first player in declaring partnership to bid this denomination
    for i, bid_id in enumerate(auction):
        if 55 <= bid_id <= 89:
            bid_denom = (bid_id - 55) % 5
            bidder = (dealer + i) % 4
            if bid_denom == denomination and bidder in (last_bidder, partner):
                return bidder

    return last_bidder


def _detect_dealer(
    hands: Dict[int, List[int]],
    auction: List[int],
    play: List[int],
) -> int:
    """
    Detect the dealer by checking which dealer assignment makes the
    opening lead consistent with the leader's hand.

    The opening leader sits to the left of the declarer.
    """
    if not play or not auction:
        return 0  # Default to North

    first_play_card = play[0]

    for candidate_dealer in range(4):
        declarer = _find_declarer(auction, candidate_dealer)
        if declarer is None:
            # All passed — leader would be left of dealer
            leader = (candidate_dealer + 1) % 4
        else:
            leader = (declarer + 1) % 4

        if first_play_card in hands[leader]:
            return candidate_dealer

    return 0  # Fallback


def _format_auction(bid_ids: List[int]) -> str:
    """Convert bid IDs to space-separated bid string."""
    if not bid_ids:
        return ""
    bids = [get_bid_from_id(bid_id) for bid_id in bid_ids]
    return " ".join(bids)


# ── Main numeric format loader ───────────────────────────────────────

def _load_numeric_format(
    path: str,
    n: Optional[int],
    mode: str = "all_bids",
) -> List[Tuple[str, str, str]]:
    """
    Load Bridge data from OpenSpiel numeric format.

    Modes
    -----
    - "all_bids": one test case per bid in the auction
    - "first_non_pass": one test case per game (first non-pass bid)
    - "last_non_pass": one test case per game (last non-pass bid)
    """
    records = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            try:
                numbers = list(map(int, line.split()))
            except ValueError:
                continue

            if len(numbers) < 56:
                continue

            deal, auction, play = _split_game_line(numbers)

            if len(deal) != 52 or not auction:
                continue

            hands = _parse_deal_interleaved(deal)
            dealer = _detect_dealer(hands, auction, play)

            if mode == "all_bids":
                for bid_idx, bid_id in enumerate(auction):
                    player = (dealer + bid_idx) % 4
                    hand_str = _decode_hand(hands[player])
                    context_bids = auction[:bid_idx]
                    auction_str = _format_auction(context_bids)
                    answer_bid = get_bid_from_id(bid_id)
                    if answer_bid.startswith("?"):
                        continue
                    records.append((hand_str, auction_str, answer_bid))

                    if n and len(records) >= n:
                        break

            elif mode == "first_non_pass":
                for bid_idx, bid_id in enumerate(auction):
                    if bid_id != 52:  # Not Pass
                        player = (dealer + bid_idx) % 4
                        hand_str = _decode_hand(hands[player])
                        context_bids = auction[:bid_idx]
                        auction_str = _format_auction(context_bids)
                        answer_bid = get_bid_from_id(bid_id)
                        if not answer_bid.startswith("?"):
                            records.append((hand_str, auction_str, answer_bid))
                        break

            elif mode == "last_non_pass":
                for bid_idx in range(len(auction) - 1, -1, -1):
                    if auction[bid_idx] != 52:
                        player = (dealer + bid_idx) % 4
                        hand_str = _decode_hand(hands[player])
                        context_bids = auction[:bid_idx]
                        auction_str = _format_auction(context_bids)
                        answer_bid = get_bid_from_id(auction[bid_idx])
                        if not answer_bid.startswith("?"):
                            records.append((hand_str, auction_str, answer_bid))
                        break

            if n and len(records) >= n:
                break

    return records


# ── Card decoding ────────────────────────────────────────────────────

def _decode_hand(card_ids: List[int]) -> str:
    """
    Convert card IDs to hand string representation.

    Card ID mapping: suit = card_id // 13 (C=0, D=1, H=2, S=3),
    rank = card_id % 13 (2=0, ..., A=12).
    """
    by_suit = {"S": [], "H": [], "D": [], "C": []}

    for card_id in card_ids:
        card_str = _id2card(card_id)
        suit, rank = card_str[0], card_str[1]
        by_suit[suit].append(rank)

    rank_order = "23456789TJQKA"
    hand_parts = []

    for suit in "SHDC":
        ranks = sorted(by_suit[suit], key=rank_order.index, reverse=True)
        hand_parts.append(f"{suit}:{''.join(ranks)}")

    return " ".join(hand_parts)


def _id2card(card_id: int) -> str:
    """Convert card ID to string representation (e.g., 0 -> 'C2', 51 -> 'SA')."""
    ranks = "23456789TJQKA"
    suits = "CDHS"
    return suits[card_id // 13] + ranks[card_id % 13]
