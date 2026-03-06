"""
Command-line interface for playing interactive Bridge games.

This module provides a CLI for playing Bridge with LLMs or human players.
"""

import asyncio
import argparse
from typing import Dict, Any, Optional

from .bridge_game import BridgeGame, Position
from .tournament import BridgeTournament


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for Bridge CLI."""
    parser = argparse.ArgumentParser(
        description="Play Bridge with LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single game command
    game_parser = subparsers.add_parser("game", help="Play a single game")
    game_parser.add_argument(
        "--convention",
        choices=["SAYC", "2/1"],
        default="SAYC",
        help="Bridge convention to use"
    )
    game_parser.add_argument(
        "--north",
        default="gpt-4o",
        help="Model for North (or 'human')"
    )
    game_parser.add_argument(
        "--south",
        default="gpt-4o",
        help="Model for South (or 'human')"
    )
    game_parser.add_argument(
        "--east",
        default="gpt-4o",
        help="Model for East (or 'human')"
    )
    game_parser.add_argument(
        "--west",
        default="gpt-4o",
        help="Model for West (or 'human')"
    )
    game_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for dealing"
    )
    
    # Tournament command
    tournament_parser = subparsers.add_parser("tournament", help="Run a tournament")
    tournament_parser.add_argument(
        "--name",
        default="LLM_Tournament",
        help="Tournament name"
    )
    tournament_parser.add_argument(
        "--convention",
        choices=["SAYC", "2/1"],
        default="SAYC",
        help="Bridge convention to use"
    )
    tournament_parser.add_argument(
        "--boards",
        type=int,
        default=8,
        help="Boards per match"
    )
    tournament_parser.add_argument(
        "--teams",
        nargs="+",
        help="Team definitions (format: 'TeamName:north_model,south_model')"
    )
    
    return parser


def parse_player_config(player_str: str) -> Dict[str, Any]:
    """Parse player configuration from string."""
    if player_str.lower() == "human":
        return {"type": "human"}
    else:
        return {"type": "llm", "model": player_str}


async def play_single_game(args: argparse.Namespace) -> None:
    """Play a single Bridge game."""
    # Setup players
    players = {
        Position.NORTH: parse_player_config(args.north),
        Position.SOUTH: parse_player_config(args.south),
        Position.EAST: parse_player_config(args.east),
        Position.WEST: parse_player_config(args.west),
    }
    
    # Create game
    game = BridgeGame(
        convention=args.convention,
        players=players
    )
    
    # Deal cards
    print(f"\n{'='*60}")
    print(f"Bridge Game - Convention: {args.convention}")
    print(f"{'='*60}")
    
    game.deal(seed=args.seed)
    
    # Show hands
    print("\nDealt hands:")
    for pos in Position:
        print(f"{pos.value}: {game.hands[pos].to_string()}")
        print(f"  HCP: {game.hands[pos].high_card_points()}")
    
    print(f"\nDealer: {game.dealer.value}")
    print(f"\n{'='*60}")
    print("Starting auction...")
    print(f"{'='*60}\n")
    
    # Play auction
    result = await game.play_auction()
    
    # Show results
    print(f"\n{'='*60}")
    print("Auction complete!")
    print(f"{'='*60}")
    print(f"Final auction: {result['auction']}")
    
    if result['contract'] != "Pass":
        print(f"Contract: {result['contract']}")
        print(f"Declarer: {result['declarer']}")
    else:
        print("All passed - no contract")
    
    # Show analysis
    analysis = game.analyze_bidding()
    print(f"\nBidding analysis:")
    print(f"Total bids: {analysis['total_bids']}")
    print(f"Passes: {analysis['passes']}")
    print(f"Suit bids: {analysis['suit_bids']}")
    
    print(f"\nBy position:")
    for pos, data in analysis['by_position'].items():
        print(f"  {pos}: {data['total']} bids, {data['hcp']} HCP")


async def run_tournament(args: argparse.Namespace) -> None:
    """Run a Bridge tournament."""
    # Create tournament
    tournament = BridgeTournament(
        name=args.name,
        convention=args.convention,
        boards_per_match=args.boards
    )
    
    # Parse and add teams
    if args.teams:
        for team_def in args.teams:
            # Format: "TeamName:north_model,south_model"
            parts = team_def.split(":")
            if len(parts) != 2:
                print(f"Invalid team format: {team_def}")
                continue
                
            team_name = parts[0]
            models = parts[1].split(",")
            if len(models) != 2:
                print(f"Invalid models for team {team_name}")
                continue
            
            tournament.create_llm_team(team_name, models[0], models[1])
    else:
        # Default teams
        tournament.create_llm_team("GPT-4 Pure", "gpt-4o", "gpt-4o")
        tournament.create_llm_team("Claude Pure", "claude-3-opus", "claude-3-opus")
        tournament.create_llm_team("GPT-Claude Mix", "gpt-4o", "claude-3-opus")
        tournament.create_llm_team("Claude-GPT Mix", "claude-3-opus", "gpt-4o")
    
    # Run tournament
    await tournament.run_tournament()


async def main():
    """Main entry point for Bridge CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "game":
        await play_single_game(args)
    elif args.command == "tournament":
        await run_tournament(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())