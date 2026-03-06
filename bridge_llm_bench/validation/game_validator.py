"""
Validation system to check if the game produces correct bids from the test dataset.

This module allows verifying that the 4-player game system produces the same
bids as expected in the test data.
"""

import asyncio
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
import csv
from datetime import datetime

from ..parsers.data_loader import load_dataset
from ..parsers.bid_parser import parse_bid_from_response, get_bid_from_id
from ..game.bridge_game import BridgeGame, Position, Bid, Auction
from ..utils.config import CONVENTIONS


class GameValidator:
    """
    Validates game bidding against test dataset.
    
    This class takes test cases from the dataset and runs them through
    the full 4-player game to verify the bidding matches expectations.
    
    Parameters
    ----------
    dataset_path : Path or str
        Path to the test dataset
    convention : str
        Bridge convention to use
    models : dict
        Model configuration for each position
    output_dir : Path, optional
        Directory for validation results
    """
    
    def __init__(
        self,
        dataset_path: Path,
        convention: str = "SAYC",
        models: Optional[Dict[Position, str]] = None,
        output_dir: Optional[Path] = None
    ):
        self.dataset_path = Path(dataset_path)
        self.convention = convention
        self.output_dir = output_dir or Path(f"validation_{datetime.now():%Y%m%d_%H%M%S}")
        
        # Default models if not specified
        if models is None:
            self.models = {
                Position.NORTH: "gpt-4o",
                Position.SOUTH: "gpt-4o",
                Position.EAST: "gpt-4o",
                Position.WEST: "gpt-4o"
            }
        else:
            self.models = models
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_test_case(
        self,
        hand: str,
        auction: str,
        expected_bid: str
    ) -> Dict[str, Any]:
        """
        Parse a test case to extract game setup information.
        
        Parameters
        ----------
        hand : str
            The hand string (South's hand in test data)
        auction : str
            The auction so far
        expected_bid : str
            The expected next bid
            
        Returns
        -------
        dict
            Parsed test case information
        """
        # Parse auction to determine dealer and current position
        bids = auction.split() if auction else []
        
        # Determine dealer (we'll assume North for simplicity)
        # In real data, dealer rotates, but we need more info to determine
        dealer = Position.NORTH
        
        # Count bids to determine whose turn it is
        num_bids = len(bids)
        positions = [Position.NORTH, Position.EAST, Position.SOUTH, Position.WEST]
        current_position = positions[num_bids % 4]
        
        # The test hand is for the player who should bid next
        test_position = current_position
        
        return {
            "hand": hand,
            "auction": auction,
            "expected_bid": expected_bid,
            "dealer": dealer,
            "test_position": test_position,
            "bids": bids
        }
    
    async def validate_single_case(
        self,
        test_case: Dict[str, Any],
        case_index: int
    ) -> Dict[str, Any]:
        """
        Validate a single test case.
        
        Parameters
        ----------
        test_case : dict
            Parsed test case
        case_index : int
            Index of the test case
            
        Returns
        -------
        dict
            Validation result
        """
        # Create a game with the test setup
        game = BridgeGame(
            convention=self.convention,
            players={
                pos: {"type": "llm", "model": self.models[pos]}
                for pos in Position
            },
            dealer=test_case["dealer"]
        )
        
        # We need to set up the game state to match the test case
        # This is tricky because we only have one hand from the test data
        
        # For now, we'll deal random cards and replace the test player's hand
        game.deal(seed=case_index)
        
        # Parse and set the test hand
        test_position = test_case["test_position"]
        # Note: This is a simplified approach. In reality, we'd need to
        # properly parse the hand string and create the Hand object
        
        # Replay the auction up to the test point
        game.auction = Auction(dealer=test_case["dealer"])
        
        position = test_case["dealer"]
        for bid_str in test_case["bids"]:
            bid = Bid.from_string(bid_str)
            game.auction.add_bid(position, bid)
            position = position.next()
        
        # Get the bid from the test position
        prompt = game.get_player_prompt(test_position)
        
        # Get LLM's bid
        try:
            actual_bid = await game.get_bid_from_player(test_position)
            actual_bid_str = str(actual_bid)
            
            # Compare with expected
            is_correct = actual_bid_str.upper() == test_case["expected_bid"].upper()
            
            result = {
                "case_index": case_index,
                "test_position": test_position.value,
                "hand": test_case["hand"],
                "auction": test_case["auction"],
                "expected_bid": test_case["expected_bid"],
                "actual_bid": actual_bid_str,
                "is_correct": is_correct,
                "model": self.models[test_position],
                "prompt": prompt,
                "success": True
            }
            
        except Exception as e:
            result = {
                "case_index": case_index,
                "test_position": test_position.value,
                "hand": test_case["hand"],
                "auction": test_case["auction"],
                "expected_bid": test_case["expected_bid"],
                "actual_bid": "ERROR",
                "is_correct": False,
                "error": str(e),
                "success": False
            }
        
        return result
    
    async def validate_dataset(
        self,
        n_cases: Optional[int] = None,
        start_index: int = 0
    ) -> Dict[str, Any]:
        """
        Validate multiple test cases from the dataset.
        
        Parameters
        ----------
        n_cases : int, optional
            Number of cases to validate (None for all)
        start_index : int
            Starting index in the dataset
            
        Returns
        -------
        dict
            Validation summary and results
        """
        # Load dataset
        print(f"Loading dataset from {self.dataset_path}")
        records = load_dataset(self.dataset_path, n_records=n_cases)
        
        if start_index > 0:
            records = records[start_index:]
        
        total_cases = len(records)
        results = []
        correct_count = 0
        
        print(f"\nValidating {total_cases} test cases")
        print(f"Convention: {self.convention}")
        print(f"Models: {self.models}")
        print("=" * 60)
        
        # Process each test case
        for i, (hand, auction, expected_bid) in enumerate(records):
            print(f"\rValidating case {i+1}/{total_cases}...", end="")
            
            # Parse test case
            test_case = self.parse_test_case(hand, auction, expected_bid)
            
            # Validate
            result = await self.validate_single_case(test_case, start_index + i)
            results.append(result)
            
            if result.get("is_correct", False):
                correct_count += 1
            
            # Save individual result
            self._save_case_result(result)
        
        print()  # New line after progress
        
        # Calculate summary statistics
        accuracy = correct_count / total_cases if total_cases > 0 else 0
        
        summary = {
            "total_cases": total_cases,
            "correct": correct_count,
            "incorrect": total_cases - correct_count,
            "accuracy": accuracy,
            "convention": self.convention,
            "models": self.models,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary and detailed results
        self._save_validation_summary(summary, results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Validation Summary:")
        print(f"Total cases: {total_cases}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print("=" * 60)
        
        return {
            "summary": summary,
            "results": results
        }
    
    def _save_case_result(self, result: Dict[str, Any]) -> None:
        """Save individual case result."""
        case_file = self.output_dir / f"case_{result['case_index']:04d}.json"
        with case_file.open("w") as f:
            json.dump(result, f, indent=2)
    
    def _save_validation_summary(
        self,
        summary: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> None:
        """Save validation summary and results."""
        # Save summary JSON
        summary_file = self.output_dir / "validation_summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed CSV
        csv_file = self.output_dir / "validation_results.csv"
        with csv_file.open("w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        # Save error analysis
        errors = [r for r in results if not r.get("is_correct", False)]
        if errors:
            error_file = self.output_dir / "validation_errors.json"
            with error_file.open("w") as f:
                json.dump(errors, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir}")


class SimplifiedValidator:
    """
    Simplified validator that directly tests bid prediction without full game simulation.
    
    This is closer to the original benchmarking approach but uses the game's
    prompt generation for consistency.
    """
    
    def __init__(
        self,
        dataset_path: Path,
        convention: str = "SAYC",
        model: str = "gpt-4o"
    ):
        self.dataset_path = Path(dataset_path)
        self.convention = convention
        self.model = model
    
    async def validate(self, n_cases: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate model predictions against test data.
        
        Parameters
        ----------
        n_cases : int, optional
            Number of cases to test
            
        Returns
        -------
        dict
            Validation results
        """
        from ..clients import get_client
        from ..metrics.evaluator import build_prompt
        
        # Load dataset
        records = load_dataset(self.dataset_path, n_records=n_cases)
        
        # Create client
        client = get_client(self.model, temperature=0.0)
        
        results = []
        correct = 0
        
        print(f"Validating {len(records)} cases with {self.model}")
        
        for i, (hand, auction, expected_bid) in enumerate(records):
            # Build prompt
            prompt = build_prompt(hand, auction, self.convention)
            
            # Get prediction
            try:
                response, _ = client.get_completion(prompt)
                predicted_bid = parse_bid_from_response(response)
                
                is_correct = predicted_bid.upper() == expected_bid.upper()
                if is_correct:
                    correct += 1
                
                results.append({
                    "index": i,
                    "hand": hand,
                    "auction": auction,
                    "expected": expected_bid,
                    "predicted": predicted_bid,
                    "is_correct": is_correct,
                    "raw_response": response
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e),
                    "is_correct": False
                })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} cases... Accuracy: {correct/(i+1):.2%}")
        
        accuracy = correct / len(records) if records else 0
        
        print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{len(records)})")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(records),
            "results": results
        }


async def run_validation_example():
    """Example of running validation."""
    # Example 1: Simplified validation (faster)
    print("Running simplified validation...")
    validator = SimplifiedValidator(
        dataset_path=Path("data/open_spiel/test.txt"),
        convention="SAYC",
        model="gpt-4o"
    )
    
    results = await validator.validate(n_cases=20)
    
    # Example 2: Full game validation (more thorough but slower)
    print("\n\nRunning full game validation...")
    game_validator = GameValidator(
        dataset_path=Path("data/open_spiel/test.txt"),
        convention="SAYC",
        models={
            Position.NORTH: "gpt-4o",
            Position.SOUTH: "gpt-4o",
            Position.EAST: "gpt-4o",
            Position.WEST: "gpt-4o"
        }
    )
    
    game_results = await game_validator.validate_dataset(n_cases=10)


if __name__ == "__main__":
    asyncio.run(run_validation_example())