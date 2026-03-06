#!/usr/bin/env python3
"""
Example script showing how to play Bridge with LLMs.

This script demonstrates:
1. Single game with 4 LLM players
2. Human vs LLM game
3. Tournament between different LLM teams
"""

import asyncio
from bridge_llm_bench.game import BridgeGame, Position, BridgeTournament


async def example_single_game():
    """Example: Play a single game with 4 LLM players."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Game with 4 LLMs")
    print("="*60)
    
    # Create game with specific LLMs for each position
    game = BridgeGame(
        convention="SAYC",
        players={
            Position.NORTH: {"type": "llm", "model": "gpt-4o"},
            Position.SOUTH: {"type": "llm", "model": "gpt-4o"},
            Position.EAST: {"type": "llm", "model": "claude-3-opus"},
            Position.WEST: {"type": "llm", "model": "claude-3-opus"}
        }
    )
    
    # Deal cards with fixed seed for reproducibility
    game.deal(seed=12345)
    
    # Show the hands
    print("\nDealt hands:")
    for pos in Position:
        hand = game.hands[pos]
        print(f"{pos.value:5} ({hand.high_card_points():2} HCP): {hand.to_string()}")
    
    # Play the auction
    print(f"\nDealer: {game.dealer.value}")
    print("\nStarting auction...")
    print("-" * 40)
    
    result = await game.play_auction()
    
    # Show results
    print("\n" + "-" * 40)
    print(f"Final auction: {result['auction']}")
    
    if result['contract'] != "Pass":
        print(f"Contract: {result['contract']} by {result['declarer']}")
    else:
        print("All passed - no game")
    
    # Analyze the bidding
    analysis = game.analyze_bidding()
    print(f"\nBidding statistics:")
    print(f"- Total bids: {analysis['total_bids']}")
    print(f"- Suit bids: {analysis['suit_bids']}")
    print(f"- Passes: {analysis['passes']}")


async def example_human_vs_llm():
    """Example: Human playing with/against LLMs."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Human (South) with LLM Partner vs LLM Opponents")
    print("="*60)
    
    # Create game with human South player
    game = BridgeGame(
        convention="SAYC",
        players={
            Position.NORTH: {"type": "llm", "model": "gpt-4o"},      # Partner
            Position.SOUTH: {"type": "human"},                        # You
            Position.EAST: {"type": "llm", "model": "claude-3-opus"}, # Opponent
            Position.WEST: {"type": "llm", "model": "claude-3-opus"}  # Opponent
        }
    )
    
    # Deal cards
    game.deal(seed=54321)
    
    # The game will prompt for human input when it's South's turn
    result = await game.play_auction()
    
    print("\n" + "-" * 40)
    print(f"Final result: {result}")


async def example_tournament():
    """Example: Tournament between different LLM teams."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Mini Tournament")
    print("="*60)
    
    # Create tournament
    tournament = BridgeTournament(
        name="LLM_Mini_Championship",
        convention="SAYC",
        boards_per_match=2  # Just 2 boards for quick demo
    )
    
    # Add teams with different LLM combinations
    tournament.create_llm_team("GPT-4 United", "gpt-4o", "gpt-4o")
    tournament.create_llm_team("Claude Alliance", "claude-3-opus", "claude-3-opus")
    tournament.create_llm_team("Mixed Masters", "gpt-4o", "claude-3-opus")
    
    # Run the tournament
    await tournament.run_tournament()
    
    print(f"\nTournament results saved to: {tournament.output_dir}")


async def example_advanced_game():
    """Example: Advanced game showing all features."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Game Analysis")
    print("="*60)
    
    # Create game
    game = BridgeGame(
        convention="2/1",  # Using 2/1 system
        players={
            Position.NORTH: {"type": "llm", "model": "gpt-4o"},
            Position.SOUTH: {"type": "llm", "model": "gpt-4o"},
            Position.EAST: {"type": "llm", "model": "gemini-1.5-pro"},
            Position.WEST: {"type": "llm", "model": "gemini-1.5-pro"}
        },
        dealer=Position.EAST  # East deals
    )
    
    # Deal a specific seed
    game.deal(seed=999)
    
    # Show detailed hand analysis
    print("\nHand Analysis:")
    for pos in Position:
        hand = game.hands[pos]
        print(f"\n{pos.value}:")
        print(f"  Hand: {hand.to_string()}")
        print(f"  HCP: {hand.high_card_points()}")
        print(f"  Distribution: ", end="")
        for suit in ["S", "H", "D", "C"]:
            count = sum(1 for card in hand.cards if card.suit.value == suit)
            print(f"{suit}={count} ", end="")
        print()
    
    # Play auction with detailed logging
    print("\nAuction (with reasoning):")
    print("-" * 60)
    
    result = await game.play_auction()
    
    # Detailed analysis
    analysis = game.analyze_bidding()
    print("\nDetailed Bidding Analysis:")
    
    for pos_name, pos_data in analysis['by_position'].items():
        print(f"\n{pos_name}:")
        print(f"  Total bids: {pos_data['total']}")
        print(f"  Passes: {pos_data['passes']}")
        print(f"  HCP: {pos_data['hcp']}")
        print(f"  Shape: {pos_data['distribution']}")


async def main():
    """Run all examples."""
    # Choose which examples to run
    examples = [
        ("Single LLM Game", example_single_game),
        # ("Human vs LLM", example_human_vs_llm),  # Commented out - requires human input
        # ("Tournament", example_tournament),       # Commented out - takes longer
        ("Advanced Analysis", example_advanced_game),
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
        
        # Pause between examples
        print("\nPress Enter to continue...")
        input()


if __name__ == "__main__":
    print("Bridge LLM Examples")
    print("==================")
    print("\nMake sure you have API keys set for the models you want to use:")
    print("- OPENAI_API_KEY for GPT models")
    print("- ANTHROPIC_API_KEY for Claude models")
    print("- GOOGLE_API_KEY for Gemini models")
    
    asyncio.run(main())