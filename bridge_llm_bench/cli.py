"""
Command-line interface for Bridge LLM Benchmarking System.

This module provides the CLI for running benchmarks on various LLMs.
"""

import argparse
import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional
import os

# Carica il file .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

import pandas as pd

from .parsers.data_loader import load_dataset, ensure_default_dataset, ensure_train_dataset
from .metrics.evaluator import evaluate
from .utils.config import ARENA_LEADERBOARD_MODELS, CONVENTIONS


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs on Contract Bridge bidding tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(ensure_default_dataset()),
        help="Path to the dataset file (textual or numeric format)."
    )
    
    parser.add_argument(
        "--models",
        nargs="*",
        help="List of models to benchmark (e.g., gpt-4o claude-3-opus-20240229)."
    )
    
    parser.add_argument(
        "--arena",
        action="store_true",
        help="Use pre-defined list of top models from LMSys Arena leaderboard."
    )
    
    parser.add_argument(
        "--n_boards",
        type=int,
        default=100,
        help="Number of boards (records) to evaluate. 0 for all."
    )
    
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/benchmark_summary.csv"),
        help="Path to save the summary CSV results."
    )
    
    parser.add_argument(
        "--log_jsonl",
        type=Path,
        default=None,
        help="Save detailed per-record logs to this JSONL file."
    )
    
    parser.add_argument(
        "--log_confusion",
        type=Path,
        default=None,
        help="Save combined confusion matrix to this CSV file."
    )
    
    parser.add_argument(
        "--log_records_csv",
        type=Path,
        default=None,
        help="Save per-record CSV logs to this file."
    )

    parser.add_argument(
        "--download-train",
        action="store_true",
        help="Download the large OpenSpiel training set (1M games, ~400MB) and exit."
    )

    parser.add_argument(
        "--mode",
        choices=["all_bids", "first_non_pass", "last_non_pass"],
        default="all_bids",
        help="Test case extraction mode for numeric datasets."
    )

    parser.add_argument(
        "--prompt_style",
        choices=["standard", "knowledge"],
        default="standard",
        help="Prompt style: 'standard' = baseline, 'knowledge' = includes SAYC reference guide."
    )

    parser.add_argument(
        "--conventions",
        nargs="*",
        choices=list(CONVENTIONS.keys()),
        default=None,
        help="Conventions to test (default: all). E.g., --conventions SAYC"
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
        
    Raises
    ------
    SystemExit
        If arguments are invalid
    """
    if not args.arena and not args.models:
        sys.exit("No models specified. Use --models or --arena.")
    
    if not args.dataset.exists():
        sys.exit(f"Dataset file not found: {args.dataset}")


def create_output_directories(args: argparse.Namespace) -> None:
    """
    Create necessary output directories.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    for path_attr in ["output_csv", "log_jsonl", "log_confusion", "log_records_csv"]:
        path = getattr(args, path_attr)
        if path and path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)


def run_benchmarks(
    records: list,
    models: list,
    conventions: dict,
    args: argparse.Namespace
) -> tuple:
    """
    Run benchmarks for all model-convention combinations.
    
    Parameters
    ----------
    records : list
        List of Bridge records to evaluate
    models : list
        List of model names to benchmark
    conventions : dict
        Dictionary of convention names and descriptions
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    tuple
        (all_results, global_confusion) containing benchmark results
    """
    all_results = []
    global_confusion = defaultdict(lambda: defaultdict(int))
    
    # Setup per-record CSV logging if requested
    record_file = None
    record_writer = None
    if args.log_records_csv:
        record_file = args.log_records_csv.open("w", newline="", encoding="utf-8")
        record_writer = csv.writer(record_file)
        record_writer.writerow([
            "index", "model", "convention", "hand", "auction",
            "reference_bid", "predicted_bid", "is_correct",
            "bridge_score", "raw_response", "latency_ms",
            "prompt_tokens", "completion_tokens"
        ])
    
    try:
        # Run evaluation for each convention and model
        for convention in conventions:
            for model in models:
                summary, conf_matrix = evaluate(
                    records,
                    model,
                    convention,
                    args.log_jsonl,
                    record_writer,
                    prompt_style=args.prompt_style,
                )
                
                if summary:  # Only append if evaluation was successful
                    all_results.append(summary)
                    
                    # Update global confusion matrix
                    for ref, preds in conf_matrix.items():
                        for pred, count in preds.items():
                            global_confusion[ref][pred] += count
    finally:
        if record_file:
            record_file.close()
    
    return all_results, global_confusion


def save_results(
    all_results: list,
    global_confusion: dict,
    args: argparse.Namespace
) -> None:
    """
    Save benchmark results to files.
    
    Parameters
    ----------
    all_results : list
        List of summary dictionaries
    global_confusion : dict
        Global confusion matrix
    args : argparse.Namespace
        Command-line arguments
    """
    if not all_results:
        print("\nNo results were generated. Check API keys and model names.")
        return
    
    # Create DataFrame and save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nBenchmark summary saved to {args.output_csv}")
    
    # Print leaderboard
    print_leaderboard(results_df)
    
    # Save confusion matrix if requested
    if args.log_confusion:
        save_confusion_matrix(global_confusion, args.log_confusion)


def print_leaderboard(results_df: pd.DataFrame) -> None:
    """
    Print formatted leaderboard to console.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with benchmark results
    """
    print("\n" + "="*25 + " LEADERBOARD " + "="*25)
    
    sorted_df = results_df.sort_values(
        by=["convention", "accuracy"],
        ascending=[True, False]
    )
    
    print(sorted_df.to_string(index=False))
    print("="*63)


def save_confusion_matrix(
    confusion: dict,
    output_path: Path
) -> None:
    """
    Save confusion matrix to CSV file.
    
    Parameters
    ----------
    confusion : dict
        Confusion matrix data
    output_path : Path
        Path to save the CSV file
    """
    # Get all unique bids
    all_bids = sorted(
        confusion.keys() | 
        {p for preds in confusion.values() for p in preds}
    )
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["reference\\predicted"] + all_bids)
        
        # Write rows
        for ref_bid in all_bids:
            row = [ref_bid] + [
                confusion[ref_bid].get(pred_bid, 0) 
                for pred_bid in all_bids
            ]
            writer.writerow(row)
    
    print(f"Combined confusion matrix saved to {output_path}")


def main():
    """
    Main entry point for the CLI.
    
    This function:
    1. Parses command-line arguments
    2. Loads the dataset
    3. Runs benchmarks for all model-convention combinations
    4. Saves and displays results
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle download-train
    if args.download_train:
        ensure_train_dataset()
        return

    # Validate arguments
    validate_args(args)

    # Load dataset
    n_records = args.n_boards if args.n_boards > 0 else None
    records = load_dataset(args.dataset, n_records, mode=args.mode)
    
    if not records:
        sys.exit(
            f"Could not load any valid records from {args.dataset}. "
            f"Please check file format or download a fresh copy."
        )
    
    print(f"Loaded {len(records)} records from {args.dataset}")
    
    # Determine models to run
    models_to_run = ARENA_LEADERBOARD_MODELS if args.arena else args.models
    
    # Create output directories
    create_output_directories(args)
    
    # Determine conventions
    if args.conventions:
        conventions_to_run = {k: CONVENTIONS[k] for k in args.conventions}
    else:
        conventions_to_run = CONVENTIONS

    # Run benchmarks
    all_results, global_confusion = run_benchmarks(
        records,
        models_to_run,
        conventions_to_run,
        args
    )
    
    # Save and display results
    save_results(all_results, global_confusion, args)


if __name__ == "__main__":
    main()