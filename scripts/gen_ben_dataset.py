"""
Generate a dataset of test positions using Ben SAYC as the oracle.
Reads hands from OpenSpiel, queries Ben SAYC for the correct bid.

Usage:
    python scripts/gen_ben_dataset.py --n_games 20 --port 8086
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge_llm_bench.parsers.data_loader import load_dataset


BEN_URL_TEMPLATE = "http://localhost:{port}"


def hand_to_ben(hand_str: str) -> str:
    """Convert 'S:AKT64 H:K D:984 C:QT76' to 'AKT64.K.984.QT76' (S.H.D.C)"""
    suits = {}
    for part in hand_str.split():
        if ':' in part:
            s, c = part.split(':', 1)
            suits[s] = c if c else ''
    return f"{suits.get('S','')}.{suits.get('H','')}.{suits.get('D','')}.{suits.get('C','')}"


def auction_to_ctx(auction_str: str) -> str:
    """Convert 'Pass Pass 1S 1NT' to Ben context '--1S1N'"""
    if not auction_str or auction_str.strip() == '':
        return ''
    parts = []
    for bid in auction_str.strip().split():
        b = bid.upper()
        if b == 'PASS':
            parts.append('--')
        elif b == 'X':
            parts.append('Db')
        elif b == 'XX':
            parts.append('Rd')
        elif b.endswith('NT'):
            parts.append(bid[0] + 'N')
        else:
            parts.append(bid[:2])
    return ''.join(parts)


def get_seat(auction_str: str) -> str:
    if not auction_str or auction_str.strip() == '':
        return 'N'
    n = len(auction_str.strip().split())
    return ['N', 'E', 'S', 'W'][n % 4]


def normalize_ben_bid(bid: str) -> str:
    b = bid.upper()
    if b in ('PASS', '--'):
        return 'Pass'
    if b in ('X', 'DB'):
        return 'X'
    if b in ('XX', 'RD'):
        return 'XX'
    if len(bid) == 2 and bid[1].upper() == 'N':
        return bid[0] + 'NT'
    return b


def query_ben(hand, auction, port):
    params = {
        'hand': hand_to_ben(hand),
        'seat': get_seat(auction),
        'dealer': 'N',
        'vul': '',
        'ctx': auction_to_ctx(auction),
    }
    try:
        r = requests.get(f"{BEN_URL_TEMPLATE.format(port=port)}/bid",
                         params=params, timeout=30)
        data = r.json()
        return normalize_ben_bid(data.get('bid', '?'))
    except Exception as e:
        return f"ERR:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=20)
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--output', default='data/ben_sayc_test.csv')
    parser.add_argument('--dataset', default='data/open_spiel/test.txt')
    args = parser.parse_args()

    # Load positions from OpenSpiel
    print(f"Loading positions from {args.dataset} ({args.n_games} games)...")
    records = load_dataset(args.dataset, n_records=args.n_games * 15, mode="all_bids")
    print(f"  Loaded {len(records)} positions from dataset")

    # Query Ben for each position
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hand', 'auction', 'wbridge5_bid', 'ben_sayc_bid'])

        for i, (hand, auction, wb5_bid) in enumerate(records):
            print(f"\r  Querying Ben: {i+1}/{len(records)}", end='', flush=True)
            ben_bid = query_ben(hand, auction, args.port)
            writer.writerow([hand, auction, wb5_bid, ben_bid])
            time.sleep(0.05)

    print(f"\n  Saved {len(records)} positions to {out_path}")

    # Quick stats
    n_match = sum(1 for h, a, w, b in
                  csv.reader(open(out_path))
                  if w.upper() == b.upper())
    print(f"  WBridge5 vs Ben SAYC agreement: {n_match-1}/{len(records)}")


if __name__ == '__main__':
    main()
