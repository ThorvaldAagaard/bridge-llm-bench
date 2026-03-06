"""
Bridge-aware scoring: partial credit for bids that are close but not exact.

Scoring rubric (0.0 to 1.0):
  1.0 - Exact match
  0.7 - Same strain, 1 level apart
  0.5 - Same level, different strain
  0.4 - Same strain, 2 levels apart
  0.3 - Pass vs low-level bid (1-2 level)
  0.1 - Pass vs high-level bid (3+ level)
  0.0 - Everything else (catastrophic mismatch)
"""

from typing import Dict, List, Tuple

# Ordered list of strains for distance calculation
STRAINS = ["C", "D", "H", "S", "NT"]
STRAIN_INDEX = {s: i for i, s in enumerate(STRAINS)}

# Special bids
SPECIAL_BIDS = {"Pass", "X", "XX", "?"}


def parse_bid(bid: str) -> Tuple[int, str]:
    """
    Parse a bid into (level, strain).

    Returns (0, bid) for special bids (Pass, X, XX).
    Returns (level, strain) for normal bids like '3NT' -> (3, 'NT').
    """
    bid = bid.strip()
    if bid.upper() in {"PASS", "P"}:
        return (0, "Pass")
    if bid in {"X", "XX", "?"}:
        return (0, bid)

    # Normal bid: first char is level, rest is strain
    if len(bid) >= 2 and bid[0].isdigit():
        level = int(bid[0])
        strain = bid[1:].upper()
        if strain in STRAIN_INDEX and 1 <= level <= 7:
            return (level, strain)

    return (0, bid)


def bid_score(predicted: str, reference: str) -> float:
    """
    Compute a bridge-aware score between predicted and reference bids.

    Parameters
    ----------
    predicted : str
        The bid predicted by the model
    reference : str
        The correct reference bid

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    pred_upper = predicted.upper().strip()
    ref_upper = reference.upper().strip()

    # Exact match (case-insensitive)
    if pred_upper == ref_upper:
        return 1.0

    p_level, p_strain = parse_bid(predicted)
    r_level, r_strain = parse_bid(reference)

    # Both are special bids but different (e.g., Pass vs X)
    if p_level == 0 or r_level == 0:
        # Pass vs normal bid
        if p_strain == "Pass" or r_strain == "Pass":
            other_level = r_level if p_strain == "Pass" else p_level
            if other_level <= 2:
                return 0.3
            return 0.1
        # X vs XX, or special vs normal
        return 0.0

    # Both are normal bids
    level_diff = abs(p_level - r_level)
    same_strain = (p_strain == r_strain)

    if same_strain and level_diff == 1:
        return 0.7
    if same_strain and level_diff == 2:
        return 0.4
    if level_diff == 0 and not same_strain:
        return 0.5
    if level_diff == 1 and not same_strain:
        return 0.2

    return 0.0


def compute_bridge_scores(
    results: List[Tuple[str, str]],
) -> Dict[str, float]:
    """
    Compute aggregate bridge-aware scores.

    Parameters
    ----------
    results : list of (predicted, reference) tuples

    Returns
    -------
    dict with keys:
        - bridge_score: mean partial score (0.0-1.0)
        - exact_match: fraction of exact matches
        - n_records: number of records scored
    """
    if not results:
        return {"bridge_score": 0.0, "exact_match": 0.0, "n_records": 0}

    scores = [bid_score(pred, ref) for pred, ref in results]
    exact = sum(1 for s in scores if s == 1.0)

    return {
        "bridge_score": round(sum(scores) / len(scores), 4),
        "exact_match": round(exact / len(scores), 4),
        "n_records": len(scores),
    }
