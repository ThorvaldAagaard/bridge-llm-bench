"""
Bridge game simulation with 4 players and coordinated bidding.

This module provides a complete Bridge game simulation where each player
(potentially an LLM) sees only their own cards and bids in sequence.
"""

from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import random
from dataclasses import dataclass, field

from ..parsers.bid_parser import parse_bid_from_response, _is_valid_bid
from ..clients import get_client, BaseClient
from ..utils.config import CONVENTIONS


class Position(Enum):
    """Bridge table positions."""
    NORTH = "North"
    EAST = "East"
    SOUTH = "South"
    WEST = "West"
    
    def next(self) -> 'Position':
        """Get the next position in clockwise order."""
        positions = list(Position)
        current_idx = positions.index(self)
        return positions[(current_idx + 1) % 4]
    
    def partner(self) -> 'Position':
        """Get the partner position."""
        if self == Position.NORTH:
            return Position.SOUTH
        elif self == Position.SOUTH:
            return Position.NORTH
        elif self == Position.EAST:
            return Position.WEST
        else:
            return Position.EAST


class Suit(Enum):
    """Card suits."""
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"
    
    def __str__(self):
        return self.value


@dataclass
class Card:
    """Represents a playing card."""
    suit: Suit
    rank: str  # 2-9, T, J, Q, K, A
    
    def __str__(self):
        return f"{self.suit}{self.rank}"
    
    def __repr__(self):
        return str(self)
    
    @property
    def value(self) -> int:
        """Get numeric value for sorting (2=2, ..., A=14)."""
        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_values[self.rank]


@dataclass
class Hand:
    """Represents a player's hand of 13 cards."""
    cards: List[Card] = field(default_factory=list)
    
    def add_card(self, card: Card) -> None:
        """Add a card to the hand."""
        self.cards.append(card)
    
    def sort(self) -> None:
        """Sort cards by suit and rank (descending)."""
        self.cards.sort(key=lambda c: (c.suit.value, -c.value))
    
    def to_string(self) -> str:
        """
        Convert hand to standard Bridge notation.
        
        Returns
        -------
        str
            Hand string like "S:AKQ H:987 D:AK C:5432"
        """
        by_suit = {suit: [] for suit in Suit}
        for card in self.cards:
            by_suit[card.suit].append(card.rank)
        
        # Sort ranks in descending order
        rank_order = "23456789TJQKA"
        result = []
        
        for suit in [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
            ranks = sorted(by_suit[suit], key=rank_order.index, reverse=True)
            result.append(f"{suit.value}:{''.join(ranks)}")
        
        return " ".join(result)
    
    def count_suit(self, suit: Suit) -> int:
        """Count cards in a specific suit."""
        return sum(1 for card in self.cards if card.suit == suit)
    
    def high_card_points(self) -> int:
        """Calculate high card points (A=4, K=3, Q=2, J=1)."""
        points = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
        return sum(points.get(card.rank, 0) for card in self.cards)


@dataclass
class Bid:
    """Represents a Bridge bid."""
    level: Optional[int] = None  # 1-7, None for Pass/X/XX
    suit: Optional[str] = None   # C/D/H/S/NT, None for Pass/X/XX
    special: Optional[str] = None  # Pass/X/XX
    
    def __str__(self):
        if self.special:
            return self.special
        return f"{self.level}{self.suit}"
    
    @classmethod
    def from_string(cls, bid_str: str) -> 'Bid':
        """Create a Bid from string representation."""
        bid_str = bid_str.upper().strip()
        
        if bid_str in ["PASS", "P"]:
            return cls(special="Pass")
        elif bid_str in ["X", "DOUBLE", "DBL"]:
            return cls(special="X")
        elif bid_str in ["XX", "REDOUBLE", "RDBL"]:
            return cls(special="XX")
        else:
            # Parse level and suit
            level = int(bid_str[0])
            suit = bid_str[1:]
            return cls(level=level, suit=suit)
    
    def is_pass(self) -> bool:
        """Check if bid is Pass."""
        return self.special == "Pass"
    
    def is_double(self) -> bool:
        """Check if bid is Double."""
        return self.special == "X"
    
    def is_redouble(self) -> bool:
        """Check if bid is Redouble."""
        return self.special == "XX"
    
    def is_suit_bid(self) -> bool:
        """Check if bid is a suit/NT bid."""
        return self.level is not None


@dataclass
class Auction:
    """Represents the bidding sequence."""
    bids: List[Tuple[Position, Bid]] = field(default_factory=list)
    dealer: Position = Position.NORTH
    
    def add_bid(self, position: Position, bid: Bid) -> None:
        """Add a bid to the auction."""
        self.bids.append((position, bid))
    
    def get_current_position(self) -> Position:
        """Get the position that should bid next."""
        if not self.bids:
            return self.dealer
        
        last_position = self.bids[-1][0]
        return last_position.next()
    
    def is_complete(self) -> bool:
        """Check if auction is complete (3 passes after a bid)."""
        if len(self.bids) < 4:
            return False
        
        # Check last 3 bids
        last_three = self.bids[-3:]
        if all(bid.is_pass() for _, bid in last_three):
            # Need at least one non-pass bid before
            return any(not bid.is_pass() for _, bid in self.bids[:-3])
        
        return False
    
    def get_last_suit_bid(self) -> Optional[Tuple[Position, Bid]]:
        """Get the last suit bid made."""
        for position, bid in reversed(self.bids):
            if bid.is_suit_bid():
                return position, bid
        return None
    
    def can_double(self, position: Position) -> bool:
        """Check if position can double."""
        last_bid = self.get_last_suit_bid()
        if not last_bid:
            return False
        
        # Can only double opponent's bid
        bid_position = last_bid[0]
        return bid_position.partner() != position
    
    def can_redouble(self, position: Position) -> bool:
        """Check if position can redouble."""
        if not self.bids:
            return False
        
        last_bid = self.bids[-1][1]
        if not last_bid.is_double():
            return False
        
        # Can only redouble opponent's double
        double_position = self.bids[-1][0]
        return double_position.partner() != position
    
    def to_string(self, from_position: Optional[Position] = None) -> str:
        """
        Convert auction to string format.
        
        Parameters
        ----------
        from_position : Position, optional
            If provided, only show bids visible to this position
            
        Returns
        -------
        str
            Space-separated bid sequence
        """
        if from_position:
            # Show bids up to but not including this position's turn
            current_pos = self.dealer
            visible_bids = []
            
            for pos, bid in self.bids:
                if pos == from_position:
                    break
                visible_bids.append(str(bid))
                current_pos = current_pos.next()
            
            return " ".join(visible_bids)
        else:
            return " ".join(str(bid) for _, bid in self.bids)


class BridgeGame:
    """
    Complete Bridge game simulation with 4 players.
    
    Parameters
    ----------
    convention : str
        Bridge convention to use ("SAYC" or "2/1")
    players : dict, optional
        Mapping of positions to player configurations
        Each value can be:
        - str: Model name for LLM player
        - dict: {"type": "llm", "model": "model-name"}
        - dict: {"type": "human"}
        - None: Use default LLM
    dealer : Position, optional
        Starting dealer position (default: North)
    """
    
    def __init__(
        self,
        convention: str = "SAYC",
        players: Optional[Dict[Position, Any]] = None,
        dealer: Position = Position.NORTH
    ):
        self.convention = convention
        self.dealer = dealer
        self.hands: Dict[Position, Hand] = {}
        self.auction = Auction(dealer=dealer)
        self.players = self._setup_players(players)
        
    def _setup_players(self, players: Optional[Dict[Position, Any]]) -> Dict[Position, Any]:
        """Setup player configurations."""
        if players is None:
            # Default all players to GPT-4
            return {pos: {"type": "llm", "model": "gpt-4o"} for pos in Position}
        
        result = {}
        for pos in Position:
            config = players.get(pos, {"type": "llm", "model": "gpt-4o"})
            
            if isinstance(config, str):
                # Just a model name
                result[pos] = {"type": "llm", "model": config}
            else:
                result[pos] = config
                
        return result
    
    def deal(self, seed: Optional[int] = None) -> None:
        """
        Deal cards to all players.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible deals
        """
        if seed is not None:
            random.seed(seed)
        
        # Create deck
        deck = []
        for suit in Suit:
            for rank in "23456789TJQKA":
                deck.append(Card(suit, rank))
        
        # Shuffle and deal
        random.shuffle(deck)
        
        # Deal 13 cards to each player
        for pos in Position:
            self.hands[pos] = Hand()
            
        for i, card in enumerate(deck):
            position = list(Position)[i % 4]
            self.hands[position].add_card(card)
        
        # Sort hands
        for hand in self.hands.values():
            hand.sort()
    
    def get_player_prompt(self, position: Position) -> str:
        """
        Build prompt for a player's bid.
        
        Parameters
        ----------
        position : Position
            The player's position
            
        Returns
        -------
        str
            Complete prompt including hand and auction
        """
        hand_str = self.hands[position].to_string()
        auction_str = self.auction.to_string(from_position=position)
        
        prompt = f"""You are playing contract bridge as {position.value}. 
{CONVENTIONS[self.convention]}

Your partner is {position.partner().value}.

Your hand: {hand_str}
Auction so far: {auction_str if auction_str else "None (you start)"}

What is your bid? Respond with a single bid (e.g., "1NT", "Pass", "X").
Consider:
- Your hand strength and distribution
- Partner's bids (if any)
- Opponents' bids (if any)
- The bidding system you're using"""
        
        return prompt
    
    async def get_bid_from_player(self, position: Position) -> Bid:
        """
        Get a bid from a player (LLM or human).
        
        Parameters
        ----------
        position : Position
            The player's position
            
        Returns
        -------
        Bid
            The player's bid
        """
        player_config = self.players[position]
        
        if player_config["type"] == "llm":
            # Get bid from LLM
            model = player_config["model"]
            client = get_client(model, temperature=0.0)
            
            prompt = self.get_player_prompt(position)
            response_text, _ = client.get_completion(prompt)
            
            bid_str = parse_bid_from_response(response_text)
            return Bid.from_string(bid_str)
            
        elif player_config["type"] == "human":
            # Get bid from human input
            print(f"\n{position.value}'s turn to bid")
            print(f"Hand: {self.hands[position].to_string()}")
            print(f"Auction: {self.auction.to_string()}")
            
            while True:
                bid_input = input("Your bid: ").strip().upper()
                if _is_valid_bid(bid_input):
                    return Bid.from_string(bid_input)
                print("Invalid bid. Please enter a valid bid (e.g., 1NT, Pass, X)")
    
    def validate_bid(self, position: Position, bid: Bid) -> bool:
        """
        Validate if a bid is legal in the current auction.
        
        Parameters
        ----------
        position : Position
            The bidding player's position
        bid : Bid
            The proposed bid
            
        Returns
        -------
        bool
            True if bid is legal
        """
        if bid.is_pass():
            return True
            
        if bid.is_double():
            return self.auction.can_double(position)
            
        if bid.is_redouble():
            return self.auction.can_redouble(position)
        
        # For suit bids, check if it's higher than the last bid
        last_suit_bid = self.auction.get_last_suit_bid()
        if last_suit_bid:
            last_bid = last_suit_bid[1]
            
            # Compare levels
            if bid.level < last_bid.level:
                return False
            elif bid.level == last_bid.level:
                # Same level - check suit rank (C < D < H < S < NT)
                suit_rank = {"C": 0, "D": 1, "H": 2, "S": 3, "NT": 4}
                return suit_rank.get(bid.suit, -1) > suit_rank.get(last_bid.suit, -1)
        
        return True
    
    async def play_auction(self) -> Dict[str, Any]:
        """
        Play through the complete auction.
        
        Returns
        -------
        dict
            Auction results including final contract and declarer
        """
        while not self.auction.is_complete():
            position = self.auction.get_current_position()
            
            # Get bid from player
            bid = await self.get_bid_from_player(position)
            
            # Validate bid
            if not self.validate_bid(position, bid):
                print(f"Invalid bid {bid} from {position.value}, defaulting to Pass")
                bid = Bid.from_string("Pass")
            
            # Add to auction
            self.auction.add_bid(position, bid)
            
            # Log the bid
            print(f"{position.value}: {bid}")
        
        # Determine final contract
        last_suit_bid = self.auction.get_last_suit_bid()
        if last_suit_bid:
            declarer_pos, contract_bid = last_suit_bid
            
            # Find first player from that partnership to bid the suit
            for pos, bid in self.auction.bids:
                if (bid.is_suit_bid() and
                    bid.suit == contract_bid.suit and
                    (pos.partner() == declarer_pos or pos == declarer_pos)):
                    declarer = pos
                    break
            else:
                declarer = declarer_pos
            
            return {
                "contract": str(contract_bid),
                "declarer": declarer.value,
                "auction": self.auction.to_string(),
                "success": True
            }
        else:
            # All passed
            return {
                "contract": "Pass",
                "declarer": None,
                "auction": self.auction.to_string(),
                "success": True
            }
    
    def analyze_bidding(self) -> Dict[str, Any]:
        """
        Analyze the completed auction for insights.
        
        Returns
        -------
        dict
            Analysis including bid quality, system adherence, etc.
        """
        analysis = {
            "total_bids": len(self.auction.bids),
            "passes": sum(1 for _, bid in self.auction.bids if bid.is_pass()),
            "suit_bids": sum(1 for _, bid in self.auction.bids if bid.is_suit_bid()),
            "doubles": sum(1 for _, bid in self.auction.bids if bid.is_double()),
            "redoubles": sum(1 for _, bid in self.auction.bids if bid.is_redouble()),
            "by_position": {}
        }
        
        # Analyze by position
        for pos in Position:
            pos_bids = [(p, b) for p, b in self.auction.bids if p == pos]
            analysis["by_position"][pos.value] = {
                "total": len(pos_bids),
                "passes": sum(1 for _, b in pos_bids if b.is_pass()),
                "hcp": self.hands[pos].high_card_points(),
                "distribution": {
                    suit.value: self.hands[pos].count_suit(suit)
                    for suit in Suit
                }
            }
        
        return analysis