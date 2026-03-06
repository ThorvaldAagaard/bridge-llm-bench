"""
Tournament system for multi-table Bridge games with different LLM combinations.

This module provides tournament functionality where different teams of LLMs
can compete against each other across multiple boards.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

from .bridge_game import BridgeGame, Position


@dataclass
class Team:
    """
    Represents a Bridge team with two players.
    
    Attributes
    ----------
    name : str
        Team name
    north_player : dict
        Configuration for North player
    south_player : dict
        Configuration for South player
    """
    name: str
    north_player: Dict[str, Any]
    south_player: Dict[str, Any]
    
    def __str__(self):
        return self.name


@dataclass
class Match:
    """
    Represents a match between two teams.
    
    Attributes
    ----------
    team_ns : Team
        North-South team
    team_ew : Team
        East-West team
    boards : List[int]
        Board numbers to play
    results : List[dict]
        Results for each board
    """
    team_ns: Team
    team_ew: Team
    boards: List[int] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_result(self, board_num: int, result: Dict[str, Any]) -> None:
        """Add a board result to the match."""
        result["board"] = board_num
        result["team_ns"] = self.team_ns.name
        result["team_ew"] = self.team_ew.name
        self.results.append(result)


class BridgeTournament:
    """
    Manages a Bridge tournament with multiple teams and boards.
    
    Parameters
    ----------
    name : str
        Tournament name
    convention : str
        Bridge convention to use
    boards_per_match : int
        Number of boards per match
    output_dir : Path, optional
        Directory for tournament output
    """
    
    def __init__(
        self,
        name: str,
        convention: str = "SAYC",
        boards_per_match: int = 8,
        output_dir: Optional[Path] = None
    ):
        self.name = name
        self.convention = convention
        self.boards_per_match = boards_per_match
        self.output_dir = output_dir or Path(f"tournaments/{name}_{datetime.now():%Y%m%d_%H%M%S}")
        self.teams: List[Team] = []
        self.matches: List[Match] = []
        self.board_seeds: Dict[int, int] = {}  # For duplicate boards
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_team(self, team: Team) -> None:
        """Add a team to the tournament."""
        self.teams.append(team)
    
    def create_llm_team(
        self,
        name: str,
        north_model: str,
        south_model: str
    ) -> Team:
        """
        Create a team with LLM players.
        
        Parameters
        ----------
        name : str
            Team name
        north_model : str
            Model name for North player
        south_model : str
            Model name for South player
            
        Returns
        -------
        Team
            The created team
        """
        team = Team(
            name=name,
            north_player={"type": "llm", "model": north_model},
            south_player={"type": "llm", "model": south_model}
        )
        self.add_team(team)
        return team
    
    def create_round_robin_schedule(self) -> List[Tuple[Team, Team]]:
        """
        Create a round-robin tournament schedule.
        
        Returns
        -------
        list
            List of (team1, team2) matchups
        """
        schedule = []
        n_teams = len(self.teams)
        
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                schedule.append((self.teams[i], self.teams[j]))
        
        return schedule
    
    async def play_match(
        self,
        team_ns: Team,
        team_ew: Team,
        board_numbers: List[int]
    ) -> Match:
        """
        Play a match between two teams.
        
        Parameters
        ----------
        team_ns : Team
            North-South team
        team_ew : Team
            East-West team
        board_numbers : List[int]
            Board numbers to play
            
        Returns
        -------
        Match
            Completed match with results
        """
        match = Match(team_ns=team_ns, team_ew=team_ew, boards=board_numbers)
        
        print(f"\nStarting match: {team_ns.name} (N/S) vs {team_ew.name} (E/W)")
        print(f"Boards: {board_numbers}")
        
        for board_num in board_numbers:
            print(f"\n--- Board {board_num} ---")
            
            # Get or generate seed for this board
            if board_num not in self.board_seeds:
                self.board_seeds[board_num] = board_num * 1000  # Deterministic seed
            
            # Setup game with player assignments
            game = BridgeGame(
                convention=self.convention,
                players={
                    Position.NORTH: team_ns.north_player,
                    Position.SOUTH: team_ns.south_player,
                    Position.EAST: team_ew.north_player,  # Team's North plays East
                    Position.WEST: team_ew.south_player   # Team's South plays West
                },
                dealer=Position(list(Position)[board_num % 4])  # Rotate dealer
            )
            
            # Deal with fixed seed for duplicate scoring
            game.deal(seed=self.board_seeds[board_num])
            
            # Play auction
            try:
                auction_result = await game.play_auction()
                
                # Analyze the bidding
                analysis = game.analyze_bidding()
                
                # Store complete result
                result = {
                    **auction_result,
                    "analysis": analysis,
                    "hands": {
                        pos.value: game.hands[pos].to_string()
                        for pos in Position
                    },
                    "dealer": game.dealer.value
                }
                
                match.add_result(board_num, result)
                
                # Save board details
                self._save_board_result(match, board_num, result)
                
            except Exception as e:
                print(f"Error on board {board_num}: {e}")
                match.add_result(board_num, {
                    "success": False,
                    "error": str(e)
                })
        
        return match
    
    async def run_tournament(self) -> None:
        """Run the complete tournament."""
        print(f"\n{'='*60}")
        print(f"Starting Tournament: {self.name}")
        print(f"Convention: {self.convention}")
        print(f"Teams: {len(self.teams)}")
        print(f"Boards per match: {self.boards_per_match}")
        print(f"{'='*60}")
        
        # Create schedule
        schedule = self.create_round_robin_schedule()
        total_matches = len(schedule)
        
        # Generate board numbers for the tournament
        total_boards = self.boards_per_match * total_matches
        board_counter = 1
        
        # Play all matches
        for match_num, (team1, team2) in enumerate(schedule, 1):
            print(f"\n{'='*40}")
            print(f"Match {match_num}/{total_matches}")
            
            # Assign board numbers for this match
            board_numbers = list(range(board_counter, board_counter + self.boards_per_match))
            board_counter += self.boards_per_match
            
            # Alternate which team sits N/S
            if match_num % 2 == 1:
                match = await self.play_match(team1, team2, board_numbers)
            else:
                match = await self.play_match(team2, team1, board_numbers)
            
            self.matches.append(match)
            
            # Save match summary
            self._save_match_summary(match)
        
        # Generate tournament report
        self._generate_tournament_report()
    
    def _save_board_result(
        self,
        match: Match,
        board_num: int,
        result: Dict[str, Any]
    ) -> None:
        """Save detailed board result to file."""
        board_file = self.output_dir / f"board_{board_num:03d}.json"
        
        board_data = {
            "board_number": board_num,
            "match": f"{match.team_ns.name} vs {match.team_ew.name}",
            "timestamp": datetime.now().isoformat(),
            **result
        }
        
        with board_file.open("w") as f:
            json.dump(board_data, f, indent=2)
    
    def _save_match_summary(self, match: Match) -> None:
        """Save match summary to file."""
        match_file = self.output_dir / f"match_{match.team_ns.name}_vs_{match.team_ew.name}.json"
        
        match_data = {
            "team_ns": match.team_ns.name,
            "team_ew": match.team_ew.name,
            "boards": match.boards,
            "results": match.results,
            "timestamp": datetime.now().isoformat()
        }
        
        with match_file.open("w") as f:
            json.dump(match_data, f, indent=2)
    
    def _generate_tournament_report(self) -> None:
        """Generate comprehensive tournament report."""
        report = {
            "tournament": self.name,
            "convention": self.convention,
            "teams": [team.name for team in self.teams],
            "total_boards": sum(len(match.boards) for match in self.matches),
            "matches": []
        }
        
        # Analyze each match
        for match in self.matches:
            match_summary = {
                "teams": f"{match.team_ns.name} vs {match.team_ew.name}",
                "boards_played": len(match.results),
                "successful_boards": sum(1 for r in match.results if r.get("success", False)),
                "contracts": []
            }
            
            for result in match.results:
                if result.get("success", False):
                    match_summary["contracts"].append({
                        "board": result["board"],
                        "contract": result.get("contract", "?"),
                        "declarer": result.get("declarer", "?")
                    })
            
            report["matches"].append(match_summary)
        
        # Save report
        report_file = self.output_dir / "tournament_report.json"
        with report_file.open("w") as f:
            json.dump(report, f, indent=2)
        
        # Generate summary statistics
        self._generate_statistics()
    
    def _generate_statistics(self) -> None:
        """Generate tournament statistics."""
        stats = []
        
        for match in self.matches:
            for result in match.results:
                if result.get("success", False):
                    board_stats = {
                        "board": result["board"],
                        "team_ns": match.team_ns.name,
                        "team_ew": match.team_ew.name,
                        "contract": result.get("contract", "?"),
                        "declarer": result.get("declarer", "?"),
                        "dealer": result.get("dealer", "?"),
                        "total_bids": result.get("analysis", {}).get("total_bids", 0)
                    }
                    
                    # Add HCP for each position
                    for pos in ["North", "South", "East", "West"]:
                        hcp = result.get("analysis", {}).get("by_position", {}).get(pos, {}).get("hcp", 0)
                        board_stats[f"{pos}_HCP"] = hcp
                    
                    stats.append(board_stats)
        
        # Create DataFrame and save
        if stats:
            df = pd.DataFrame(stats)
            df.to_csv(self.output_dir / "tournament_statistics.csv", index=False)
            
            # Generate summary
            print(f"\n{'='*60}")
            print("Tournament Summary:")
            print(f"Total boards played: {len(stats)}")
            print(f"Average bids per board: {df['total_bids'].mean():.1f}")
            print(f"\nContracts reached:")
            print(df['contract'].value_counts())
            print(f"{'='*60}")


async def run_sample_tournament():
    """Run a sample tournament with different LLM teams."""
    # Create tournament
    tournament = BridgeTournament(
        name="LLM_Championship",
        convention="SAYC",
        boards_per_match=4
    )
    
    # Create teams with different LLM combinations
    tournament.create_llm_team("GPT-4 Team", "gpt-4o", "gpt-4o")
    tournament.create_llm_team("Claude Team", "claude-3-opus", "claude-3-opus")
    tournament.create_llm_team("Mixed Team 1", "gpt-4o", "claude-3-opus")
    tournament.create_llm_team("Gemini Team", "gemini-1.5-pro", "gemini-1.5-pro")
    
    # Run tournament
    await tournament.run_tournament()
    
    print(f"\nTournament complete! Results saved to: {tournament.output_dir}")


if __name__ == "__main__":
    # Run sample tournament
    asyncio.run(run_sample_tournament())