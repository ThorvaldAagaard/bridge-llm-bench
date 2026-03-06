# Bridge LLM Bench

An open-source benchmark for evaluating how well Large Language Models can bid at contract bridge.

## What This Is

Give an LLM a bridge hand and the auction so far, ask it to make a call, and compare its answer to a reference engine. Simple in concept, but full of surprises — see the [full writeup](docs/bridgewinners_post.md) for the story.

### Key Findings

- **WBridge5 is not SAYC.** The OpenSpiel dataset uses WBridge5 bids as reference, but WBridge5 agrees with dedicated SAYC engines only ~44% of the time. We switched to Ben SAYC as oracle and scores jumped immediately.
- **Prompt engineering matters more than model size.** Gemini Flash Lite (Google's smallest model) reached **80% accuracy** with targeted prompts + majority voting, beating Gemini 3.1 Pro (70%) on the same task.
- **Chain-of-thought hurts small models.** Flash Lite drops from 54% to 46% with CoT — it generates plausible but wrong reasoning.
- **The Law of Total Tricks is a trap for LLMs.** Adding LOTT rules makes models bid too aggressively. The fix: for every "bid with X" rule, add a "don't bid with Y" rule.

### Accuracy Journey (Gemini Flash Lite on Ben SAYC)

| Stage | Accuracy |
|-------|----------|
| Baseline SAYC prompt | 54% |
| + Position-aware prompting | 62% |
| + Balanced competitive rules | 66% |
| + Targeted error examples | 72% |
| + Majority voting (k=9, t=0.5) | **80%** |

## Supported Models

The benchmark supports 11 LLM providers:

| Provider | Models | Status |
|----------|--------|--------|
| OpenAI | GPT-5.2, GPT-5.1, o3 | Working |
| Anthropic | Claude Opus 4.6, Sonnet 4.6 | Working |
| Google | Gemini 3.1 Pro, 3 Flash, Flash Lite | Working |
| DeepSeek | V3, R1 | Working |
| Zhipu AI | GLM-5, GLM-4.7 | Working |
| Qwen | Qwen 3.5 | API key needed |
| xAI | Grok 4 | API key needed |
| Baidu | ERNIE 5.0 | API key needed |
| Moonshot | Kimi K2.5 | API key needed |
| Bytedance | Dola Seed 2.0 | API key needed |

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/bridge-llm-bench.git
cd bridge-llm-bench

# Install
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your keys

# Run benchmark (example: 50 hands on Claude)
python -m bridge_llm_bench.cli --models claude-sonnet-4-6 --n_boards 50

# Run prompt optimization experiment
python scripts/optimize_prompt.py --prompt_id 18 --vote_k 9 --temperature 0.5
```

## Project Structure

```
bridge_llm_bench/
    clients/          # LLM API clients (OpenAI, Anthropic, Gemini, etc.)
    parsers/          # Dataset parsers (OpenSpiel numeric, text pipe format)
    metrics/          # Evaluation and bridge-aware scoring
    game/             # Full bridge game simulation
    utils/            # Config, pricing, prompt templates
    validation/       # Game state validation

scripts/
    optimize_prompt.py    # Prompt engineering experiments (20 strategies)
    compare_oracles.py    # Compare WBridge5 vs Ben SAYC vs saycbridge
    gen_ben_dataset.py    # Generate test positions from Ben SAYC

data/
    ben_sayc_100.csv      # 150 positions with Ben SAYC reference bids
    open_spiel/test.txt   # 10K OpenSpiel games (~110K bidding positions)

tests/                    # Test suite
```

## Datasets

### Ben SAYC (recommended for SAYC testing)
150 carefully selected positions with bids from the Ben neural network engine configured for SAYC. This is the primary test set for prompt optimization.

### OpenSpiel
10,000 games from WBridge5 self-play (~110,000 bidding positions). Large and useful for stress-testing, but the reference bids are NOT standard SAYC.

## Oracle Comparison

We compared three bridge engines on the same positions:

| Comparison | Agreement |
|------------|-----------|
| WBridge5 vs Ben SAYC | 44% |
| WBridge5 vs saycbridge | 48% |
| Ben SAYC vs saycbridge | 64% |

WBridge5 opens 4-card majors, uses different NT ranges, and makes competitive decisions that don't follow SAYC. Ben and saycbridge agree much more closely with each other.

## Prompt Optimization

The `scripts/optimize_prompt.py` framework includes 20 different prompt strategies tested on Flash Lite:

- **P0**: Basic SAYC knowledge (baseline, 54%)
- **P9**: Position-aware prompting (62%)
- **P14-P17**: LOTT experiments (56-66%)
- **P18**: Targeted error examples (72% single, **80% with voting**)
- **P19**: Aggressive competitive framing (62%)

### Majority Voting

Call the model K times with temperature > 0, take the majority answer:

| Config | Accuracy |
|--------|----------|
| Single call (t=0) | 72% |
| 5 votes (t=0.5) | 74% |
| 7 votes (t=0.5) | 78% |
| 9 votes (t=0.5) | **80%** |

## Contributing

Contributions welcome! Especially:

1. **SAYC Oracle Validation** — Review Ben SAYC's bids on the test positions
2. **Better Test Datasets** — BBO game records, club hand records, expert problem sets
3. **Scoring Improvements** — Partial credit for "close" bids vs catastrophic errors
4. **New Conventions** — 2/1 Game Forcing, Precision, ACOL, etc.
5. **New LLM Providers** — Help testing models we don't have API access for

See [docs/bridgewinners_post.md](docs/bridgewinners_post.md) for the full story and discussion.

## License

MIT
