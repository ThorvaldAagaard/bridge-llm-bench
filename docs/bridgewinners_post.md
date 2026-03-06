# Can LLMs Play SAYC? Building a Bridge Bidding Benchmark — and What I've Learned So Far

Hi everyone,

I've been working on something that I think this community might find interesting (and where I could really use your expertise). I'm building an open-source benchmark to test how well Large Language Models — ChatGPT, Claude, Gemini, DeepSeek, and others — can bid at contract bridge.

The basic setup is straightforward: you give the LLM a hand and the auction so far, and ask it to make a call. Then you compare its answer to a reference. Simple in concept, but the journey has been full of surprises.

I started with SAYC as the first convention to test, since it's well-documented and widely known. But I'd love to expand to 2/1 Game Forcing, Precision, and other popular systems — more on that later.

## Why LLMs Instead of Traditional AI?

Before diving into the results, I want to share the methodological reasoning behind this project.

Bridge already has strong AI engines built with traditional machine learning — neural networks trained on millions of hands through self-play (like Ben) or formal logic solvers (like saycbridge). These are great at what they do, and per-inference they're far cheaper than calling an LLM API. So why bother with LLMs?

The answer is **flexibility**. A traditional ML bridge bot requires months of training data collection, custom architecture design, reward shaping, and self-play pipelines. If you want it to play a different convention — say switching from SAYC to Precision — you essentially need to retrain from scratch. An LLM, on the other hand, can be instructed in natural language. You describe the convention in the prompt, give it a few examples, and it starts bidding. No retraining, no new dataset, no specialized infrastructure. It's orders of magnitude more agile.

This opens up some fascinating possibilities. You could potentially **fine-tune an LLM on a specific bidding system** — feed it a corpus of expert-annotated SAYC hands and get a model that really understands SAYC nuances. Or go even further: fine-tune on the bidding style of a specific champion pair. Imagine an LLM that bids like Meckstroth-Rodwell, or one that emulates the Italian Blue Team's approach. The "knowledge" lives in text, which is exactly what LLMs are designed to process.

The approach I've been exploring is what I'd call **knowledge injection through prompting** — instead of training the model on bridge hands, you write a detailed prompt that encodes the convention's rules, gives targeted examples, and tells the model exactly what to do in specific situations. The model already has some bridge knowledge from its general training data, and the prompt sharpens it. Think of it like giving a reasonably knowledgeable partner a very detailed system card before a session.

The cost trade-off is real — an LLM API call costs far more than running a neural network inference locally. But for benchmarking and research purposes, and especially for exploring whether LLMs can learn conventions from natural language descriptions, I think it's a worthwhile approach.

## The WBridge5 Problem

My first dataset came from [OpenSpiel](https://github.com/google-deepmind/open_spiel) — 10,000 games of WBridge5 self-play, giving about 110,000 bidding positions with reference bids. I assumed WBridge5 was playing something close to SAYC, so I used its bids as the "correct answer."

The results were puzzling. Even the best LLMs could only match about 44% of WBridge5's bids. At first I thought the models were just bad at bridge. But then I started looking at the actual bids WBridge5 was making, and something didn't add up.

So I set up two independent SAYC engines to compare:
- **[Ben](https://github.com/lorserker/ben)** — a neural network bridge engine by Lorents Samson, trained specifically on SAYC
- **[saycbridge](https://github.com/nickstenning/saycbridge)** — a Z3 theorem prover-based SAYC bidder from Google

I ran all three oracles on the same 25 positions. The results were eye-opening:

| Comparison | Agreement |
|-----------|-----------|
| WBridge5 vs Ben SAYC | 44% |
| WBridge5 vs saycbridge | 48% |
| Ben SAYC vs saycbridge | 64% |

WBridge5 agrees with the two SAYC engines less than half the time, while Ben and saycbridge agree with each other nearly two-thirds of the time. Looking at the specific disagreements, WBridge5 opens 4-card majors, uses different notrump ranges, and makes competitive decisions that clearly don't follow standard SAYC.

**Lesson learned: if you're testing an LLM's SAYC knowledge against a non-SAYC oracle, you're measuring noise, not skill.** I generated a clean dataset of 150 positions using Ben SAYC as the reference, and suddenly all the LLM scores improved dramatically — GPT went from 20% to 44%, Claude from 44% to 56%.

## Pushing a Tiny Model to 80%

With a proper oracle in place, I decided to see how far I could push prompt engineering on **Gemini 3.1 Flash Lite** — Google's smallest and cheapest model. I chose it deliberately: if you can make the weakest model perform well, the techniques should transfer to stronger models too.

The baseline was 54% accuracy with a simple SAYC knowledge prompt. Here's what I tried and what happened:

### Things That Didn't Work

**Chain-of-thought reasoning** actually made things worse — 46% instead of 54%. Flash Lite is just too small to reason productively. When you tell it to "think step by step," it generates plausible-sounding but wrong analysis. Explicit examples work much better than reasoning instructions for small models.

**The Law of Total Tricks** was an interesting experiment. I added LOTT rules to help with competitive bidding decisions (count your combined trumps, compete to the corresponding level). The result? Accuracy dropped to 56%. The model became too aggressive — it started bidding on every hand where it could count 8+ combined trumps, ignoring all the cases where passing is correct. The LOTT is a useful guideline for humans who already have good judgment about when to apply it, but for an LLM it became a blanket rule to always compete.

**Bigger models don't always help.** Gemini 3.1 Pro (the full reasoning model) scored only 70% on the same prompt, compared to 72% for Flash Lite. The reasoning model overthinks — I saw it bid 7H on a hand where 2H was correct, apparently convinced by its own chain of reasoning that the hand was worth a grand slam. Sometimes simplicity is an advantage.

### Things That Worked

**Targeted few-shot examples** gave the biggest single gain. This is the core of the knowledge injection approach: instead of generic bidding examples, I analyzed the specific errors the model was making and created examples for those exact patterns. For instance, the model kept passing with 3-card support for partner's major after an opponent's overcall. So I added:

> S:Q52 H:6543 D:K732 C:AJ (10 HCP, 3S support) | P P 1S 1NT → X
> (COMPETITIVE DOUBLE: 10 HCP + 3-card support for partner's 1S. Do NOT pass!)

Each of these targeted examples acts as a "knowledge patch" — you're not retraining the model, you're correcting specific gaps in its understanding through natural language. This kind of targeted example was worth more than pages of general rules.

The prompt also includes a structured SAYC knowledge block covering opening bid requirements, responses, NT ranges, competitive bidding rules, and slam conventions. It's essentially a condensed system card that the model reads before every hand. The key insight is that **how you structure this knowledge matters enormously** — a wall of text helps less than a well-organized set of rules with concrete examples.

**"When NOT to compete" rules** were essential to balance things out. Every time I added competitive bidding advice, the model overcorrected. The fix was always the same: for every "bid with X" rule, add a corresponding "but don't bid with Y" rule. Pass with misfit. Don't overcall at the 2-level with less than 10 HCP. After your initial Pass limited your hand, don't suddenly bid aggressively. These negative examples were just as important as the positive ones.

**Majority voting** was the most powerful single technique. You call the model 9 times with some randomness (temperature 0.5), and take the majority answer. On any given position the model might sometimes pass and sometimes bid correctly — with 9 samples, the majority usually gets it right.

| Config | Accuracy |
|--------|----------|
| Single call | 72% |
| 5 votes | 74% |
| 7 votes | 78% |
| **9 votes** | **80%** |

### The Full Journey

| Stage | Accuracy | What Changed |
|-------|----------|-------------|
| Baseline (SAYC knowledge) | 54% | Starting point |
| + Position-aware prompting | 62% | Tell model its seat position |
| + LOTT balanced (when to AND when not to compete) | 66% | Competitive rules with guardrails |
| + Targeted error examples | 72% | Few-shot for specific failure modes |
| + Majority voting (k=9) | **80%** | Consensus filtering |

## The Remaining 20%

Even with 80% accuracy, there are 10 positions (out of 50) that the model consistently gets wrong. Looking at them, they're genuinely difficult:

- Competitive raises at the 3-5 level where you need to judge "is this a game hand or a sacrifice?"
- Weak 2 openings with borderline hands (6-card suit, 8 HCP — do you open or pass?)
- Whether to bid 1NT or 3NT after an opponent's overcall
- Complex competitive sequences with multiple doubles and redoubles

Some of these would challenge many intermediate-level human players too. I'm curious whether stronger models or different prompting approaches can crack these.

## Beyond Bidding: Cardplay

Bidding is just the first half of bridge. The framework already includes a full game simulation module where LLMs can play out hands — opening lead, declarer play, defense — and I'm planning to build a cardplay benchmark next.

Cardplay is an interesting challenge for LLMs because it involves different cognitive skills than bidding: counting cards, planning sequences of plays, managing entries, executing squeezes and endplays. It's also more constrained in some ways (you can see dummy, the legal plays are obvious) but more complex in others (you need to track 52 cards across 13 tricks).

I haven't done systematic testing on cardplay yet, but early experiments suggest LLMs can follow basic suit management and sometimes even find finesses. Whether they can execute a multi-step plan like a squeeze or a dummy reversal is an open question I'm excited to explore. If anyone has ideas for how to benchmark cardplay quality — what reference engines to use, what metrics make sense — I'd love to hear them.

## Models Tested and Planned

So far the prompt optimization work has been done primarily on Gemini Flash Lite (to prove the techniques on the weakest model). Early tests on other models with the WBridge5 oracle showed:

| Model | Provider | Accuracy (WBridge5 oracle) |
|-------|----------|---------------------------|
| Claude Sonnet 4.5 | Anthropic | 44% |
| GPT-4o | OpenAI | 20% |
| Gemini Flash Lite | Google | ~30% |

These numbers jumped significantly with the Ben SAYC oracle (Claude to 56%, GPT to 44%), and I expect them to jump again with the optimized P18 prompt and voting.

The benchmark framework supports 11 LLM providers. Here's what I want to test with the best prompt:

**Working API access:**
- **Anthropic** — Claude Opus 4.6, Claude Sonnet 4.6
- **OpenAI** — GPT-5.2, GPT-5.1, o3
- **Google** — Gemini 3.1 Pro, Gemini 3 Flash, Flash Lite
- **DeepSeek** — DeepSeek-V3, DeepSeek-R1
- **Zhipu AI** — GLM-5, GLM-4.7

**Need API access / credits:**
- **xAI** — Grok 4
- **Alibaba** — Qwen 3.5
- **Baidu** — ERNIE 5.0
- **Moonshot** — Kimi K2.5
- **Bytedance** — Dola Seed 2.0

The goal is a comprehensive leaderboard showing which AI is best at bridge bidding. I'd also love to see how the reasoning models (o3, DeepSeek-R1, Claude with extended thinking) compare to their faster counterparts — does "thinking harder" actually help in bridge?

## What's Next — and Where I Need Your Help

### More Conventions

I started with SAYC because it's well-documented, but bridge is played with hundreds of different systems. This is where the LLM approach really shines — in theory, you can teach a new convention just by rewriting the prompt and knowledge block, without any retraining. I'd love to test:
- **2/1 Game Forcing** — probably the most common system among experienced players
- **Precision** — interesting because it's so different from Standard American
- **ACOL** — for the UK/European audience
- **Any other system** where we can get a reliable oracle

The benchmark framework already supports multiple conventions — it just needs the knowledge blocks and reference engines.

### Where the Community Can Help

**1. SAYC Oracle Validation.** Are Ben SAYC's bids actually good SAYC? I've looked at 150 positions and most seem reasonable, but I'm not a SAYC expert. I'd really value having experienced SAYC players review a sample of the test positions.

**2. Better Test Datasets.** The OpenSpiel games are fine for volume, but the auctions can be unusual since they come from WBridge5 self-play. I'd love access to:
- Recorded SAYC games from BBO
- Hand records from clubs/tournaments where SAYC is the agreed system
- Curated "what should I bid?" problem sets with expert answers

**3. Scoring Beyond Exact Match.** Right now scoring is binary — match the reference or don't. But in bridge, some disagreements are reasonable (2S vs 3S might be a judgment call) while others are catastrophic (7NT vs Pass). I've implemented a partial-credit system but would love input from experienced players on how to weight different types of errors.

**4. Vulnerability Information.** The OpenSpiel dataset doesn't clearly encode vulnerability, so I'm defaulting to "non-vulnerable" for all positions. This obviously affects competitive decisions. Does anyone know how to extract vulnerability from the OpenSpiel numeric format?

**5. Other Engines/Oracles.** Are there other SAYC-playing engines I should compare against? Any publicly available bridge AI that plays a known, well-defined system?

**6. Fine-tuning Data.** If the knowledge injection approach hits a ceiling, the next step would be fine-tuning an LLM on bridge-specific data. Expert-annotated hands with explanations (not just bids, but *why* that bid) would be incredibly valuable for this. Does anyone know of datasets like this?

## The Code

The project is open-source on GitHub: **[bridge-llm-bench](https://github.com/albertogerli/bridge-llm-bench)**

It includes:
- The prompt optimization framework with all 20 prompt strategies tested
- 11 LLM provider clients (OpenAI, Anthropic, Google, DeepSeek, Qwen, xAI, Zhipu, Baidu, Moonshot, Bytedance, Minimax)
- Scripts to query Ben SAYC and saycbridge as oracles
- The oracle comparison tool
- The 150-position Ben SAYC test dataset
- Evaluation and bridge-aware scoring modules

The whole thing is in Python and fairly straightforward to run if you have API access to any of the supported providers.

## Why This Matters

Bridge is an interesting benchmark for AI because it requires:
- **Knowledge** of a complex rule system (bidding conventions)
- **Inference** about partner's and opponents' hands from the auction
- **Judgment** about when to compete, when to pass, when to sacrifice
- **Communication** through a constrained protocol (the bidding system)

Unlike chess or Go where we have superhuman AI, bridge bidding is still a domain where the best AI systems (like Ben's neural models) are strong but imperfect, and where LLMs offer a fundamentally different approach — they can be instructed in natural language and can learn from examples rather than self-play.

The long-term dream is an LLM that can be told "play Precision with Kantar-style cue bids" and just... do it. We're not there yet, but 80% accuracy on a tiny model with just prompt engineering is a promising start. I think tracking how LLMs improve at bridge over time could be a fascinating window into AI progress on strategic reasoning tasks.

Looking forward to your thoughts and suggestions!

---

*P.S. — If you're interested in collaborating or have datasets/expertise to share, please reach out. This is very much a community project and I'd love to have bridge players involved in shaping how we evaluate AI at our game.*
