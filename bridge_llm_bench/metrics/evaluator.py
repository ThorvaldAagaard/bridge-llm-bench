"""
Evaluation module for benchmarking LLMs on Bridge bidding tasks.

This module provides functions to evaluate model performance, calculate metrics,
and generate confusion matrices.
"""

import time
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import csv

from ..parsers.bid_parser import parse_bid_from_response
from ..utils.config import CONVENTIONS, PROMPT_TEMPLATE, PROMPT_TEMPLATES, SAYC_KNOWLEDGE, PRICE_USD_PER_1K
from ..clients import get_client
from .bridge_scoring import bid_score


_HCP = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}


def hand_info(hand: str) -> str:
    """
    Compute HCP and distribution from a hand string.

    Parameters
    ----------
    hand : str
        e.g. "S:AKT64 H:K D:984 C:QT76"

    Returns
    -------
    str
        e.g. "13 HCP, 5=1=3=4"
    """
    suits = []
    hcp = 0
    for part in hand.split():
        cards = part.split(':')[1] if ':' in part else part
        suits.append(len(cards))
        hcp += sum(_HCP.get(c, 0) for c in cards)
    dist = '='.join(str(n) for n in suits)
    return f"{hcp} HCP, {dist}"


def build_prompt(hand: str, auction: str, convention: str,
                  prompt_style: str = "standard") -> str:
    """
    Build a prompt for the LLM to bid on a Bridge hand.

    Parameters
    ----------
    hand : str
        String representation of the hand (e.g., "S:AKQ H:987 D:AK C:5432")
    auction : str
        Space-separated sequence of bids so far (e.g., "1H Pass")
    convention : str
        Bridge convention being used ("SAYC" or "2/1")
    prompt_style : str
        Prompt style: "standard" or "knowledge"

    Returns
    -------
    str
        Complete prompt for the LLM
    """
    template = PROMPT_TEMPLATES.get(prompt_style, PROMPT_TEMPLATE)
    kwargs = dict(
        convention_details=CONVENTIONS[convention],
        hand=hand,
        auction=auction if auction else "None",
    )
    if prompt_style == "knowledge":
        kwargs["knowledge"] = SAYC_KNOWLEDGE
    return template.format(**kwargs)


def evaluate(
    records: List[Tuple[str, str, str]],
    model_name: str,
    convention: str,
    log_jsonl_path: Optional[Path] = None,
    record_writer: Optional[csv.writer] = None,
    prompt_style: str = "standard",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    """
    Evaluate a model's performance on Bridge bidding tasks.
    
    Parameters
    ----------
    records : list of tuple
        List of (hand, auction, reference_bid) tuples to evaluate
    model_name : str
        Name of the model to evaluate
    convention : str
        Bridge convention to use ("SAYC" or "2/1")
    log_jsonl_path : Path, optional
        Path to save detailed logs in JSONL format
    record_writer : csv.writer, optional
        CSV writer for per-record results
        
    Returns
    -------
    tuple
        A tuple containing:
        - summary: Dict with aggregate metrics (accuracy, tokens, cost, etc.)
        - confusion_matrix: Dict mapping reference bids to predicted bids with counts
        
    Notes
    -----
    The summary dictionary contains:
    - model: Model name
    - convention: Convention used
    - n_records: Number of records evaluated
    - accuracy: Fraction of correct predictions
    - avg_latency_ms: Average response time in milliseconds
    - prompt_tokens: Total prompt tokens used
    - completion_tokens: Total completion tokens used
    - total_tokens: Total tokens (prompt + completion)
    - estimated_cost_usd: Estimated cost based on pricing data
    - total_time_s: Total evaluation time in seconds
    
    Examples
    --------
    >>> records = [("S:AKQ H:987 D:AK C:5432", "1H Pass", "2H")]
    >>> summary, confusion = evaluate(records, "gpt-4o", "SAYC")
    >>> print(f"Accuracy: {summary['accuracy']}")
    Accuracy: 0.85
    """
    try:
        client = get_client(model_name, temperature=0.0)
    except (ValueError, ImportError) as e:
        print(f"[ERROR] Could not initialize client for {model_name}: {e}")
        # Return a summary with zeros to indicate failure
        return {
            "model": model_name,
            "convention": convention,
            "n_records": len(records),
            "accuracy": 0.0,
            "avg_latency_ms": 0.0,
            "total_time_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "error": str(e)
        }, defaultdict(lambda: defaultdict(int))
    
    total_records = len(records)
    correct_predictions = 0
    total_bridge_score = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0
    
    # Confusion matrix: {reference_bid: {predicted_bid: count}}
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    start_time = time.time()
    
    for i, (hand, auction, ref_bid) in enumerate(records):
        print(f"\r→ Evaluating {model_name} [{convention}]: {i+1}/{total_records}", end="")
        
        prompt = build_prompt(hand, auction, convention, prompt_style)
        
        latency_start = time.time()
        try:
            response_text, usage = client.get_completion(prompt)
            predicted_bid = parse_bid_from_response(response_text)
        except Exception as e:
            print(f"\n[API ERROR] Model: {model_name}, Record: {i}. Error: {e}")
            response_text, predicted_bid = f"ERROR: {e}", "?"
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        latency_ms = (time.time() - latency_start) * 1000
        
        is_correct = (predicted_bid.upper() == ref_bid.upper())
        partial = bid_score(predicted_bid, ref_bid)
        if is_correct:
            correct_predictions += 1
        total_bridge_score += partial

        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_latency_ms += latency_ms
        confusion_matrix[ref_bid][predicted_bid] += 1
        
        # Log detailed results
        if record_writer:
            record_writer.writerow([
                i,
                model_name,
                convention,
                hand,
                auction,
                ref_bid,
                predicted_bid,
                is_correct,
                round(partial, 2),
                response_text,
                round(latency_ms, 1),
                usage["prompt_tokens"],
                usage["completion_tokens"],
            ])
        
        if log_jsonl_path:
            log_entry = {
                "index": i,
                "model": model_name,
                "convention": convention,
                "hand": hand,
                "auction": auction,
                "reference_bid": ref_bid,
                "predicted_bid": predicted_bid,
                "is_correct": is_correct,
                "bridge_score": round(partial, 2),
                "raw_response": response_text,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
            }
            _append_to_jsonl(log_jsonl_path, log_entry)
    
    print()  # Newline after progress bar
    
    end_time = time.time()
    total_time_s = end_time - start_time
    
    # Calculate cost (split input/output pricing)
    price_key = next((k for k in PRICE_USD_PER_1K if model_name.startswith(k)), None)
    price = PRICE_USD_PER_1K.get(price_key, {})
    total_tokens = total_prompt_tokens + total_completion_tokens
    if isinstance(price, dict):
        estimated_cost = (
            (total_prompt_tokens / 1000) * price.get("input", 0)
            + (total_completion_tokens / 1000) * price.get("output", 0)
        )
    else:
        estimated_cost = (total_tokens / 1000) * price if price > 0 else 0.0
    
    summary = {
        "model": model_name,
        "convention": convention,
        "n_records": total_records,
        "accuracy": round(correct_predictions / total_records, 4) if total_records > 0 else 0,
        "bridge_score": round(total_bridge_score / total_records, 4) if total_records > 0 else 0,
        "avg_latency_ms": round(total_latency_ms / total_records, 1) if total_records > 0 else 0,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 4),
        "total_time_s": int(total_time_s),
    }
    
    return summary, confusion_matrix


def calculate_confusion_metrics(confusion_matrix: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """
    Calculate additional metrics from confusion matrix.
    
    Parameters
    ----------
    confusion_matrix : dict
        Confusion matrix mapping reference bids to predicted bids
        
    Returns
    -------
    dict
        Dictionary containing precision, recall, and F1 scores per bid
        
    Notes
    -----
    For each bid type, calculates:
    - precision: TP / (TP + FP)
    - recall: TP / (TP + FN)
    - f1_score: 2 * (precision * recall) / (precision + recall)
    """
    metrics = {}
    all_bids = set()
    
    # Collect all unique bids
    for ref_bid, predictions in confusion_matrix.items():
        all_bids.add(ref_bid)
        all_bids.update(predictions.keys())
    
    for bid in all_bids:
        # True positives: correctly predicted this bid
        tp = confusion_matrix.get(bid, {}).get(bid, 0)
        
        # False positives: predicted this bid when it wasn't correct
        fp = sum(
            confusion_matrix.get(ref, {}).get(bid, 0)
            for ref in all_bids
            if ref != bid
        )
        
        # False negatives: should have predicted this bid but didn't
        fn = sum(
            count
            for pred, count in confusion_matrix.get(bid, {}).items()
            if pred != bid
        )
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[bid] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "support": tp + fn,  # Total instances of this bid
        }
    
    return metrics


def _append_to_jsonl(file_path: Path, data: Dict) -> None:
    """
    Append a dictionary as a new line in a JSONL file.
    
    Parameters
    ----------
    file_path : Path
        Path to the JSONL file
    data : dict
        Dictionary to append
    """
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")