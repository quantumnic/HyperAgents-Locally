# HyperAgents-Locally

**Self-Improving AI Agents — Local, Cloud, or Free**

A fork of [facebookresearch/HyperAgents](https://github.com/facebookresearch/HyperAgents) extended with local Ollama, Apple Silicon MLX, OpenRouter cloud gateway, and a Rust port of the evolution loop.

> **Paper:** [Hyperagents](https://arxiv.org/abs/2603.19461) — Self-referential agents that integrate a task agent and a meta agent into a single editable program, enabling metacognitive self-modification.

---

## Language map

| Feature | Python | Rust |
|---|:---:|:---:|
| Evolution loop (TaskAgent + MetaAgent) | ✓ `python/loop.py` | ✓ `rust/src/runner.rs` |
| Agent-to-Agent communication loop | ✓ `python/comms/loop.py` | ✓ `rust/src/comms/` |
| LLM client (Ollama / llama.cpp / OpenRouter / MLX) | ✓ `python/agent/llm.py` | ✓ `rust/src/llm.rs` |
| Evaluation harness | ✓ `python/domains/harness.py` | ✓ `rust/src/domains/harness.rs` |
| Domains: text_classify, emotion, factory | ✓ | ✓ |
| Domains: rust, search_arena, paper_review | ✓ | — |
| Apple Silicon MLX support | ✓ (`--mlx`) | — |

**Binaries built by `bash install.sh --rust`:**
- `rust/target/release/hyperagents` — evolution loop
- `rust/target/release/hyperagents-comms` — agent communication loop

The Rust binary is intentionally scoped to the three shared benchmark domains (`text_classify`, `emotion`, `factory`). The Python harness adds the extra `search_arena` and `paper_review` suites.

---

## How it works

```
┌─────────────────────────────────────────────────────────┐
│  Hyper Loop                                             │
│                                                         │
│  Gen 0:  TaskAgent solves tasks  →  baseline score      │
│                                                         │
│  Gen N:  MetaAgent (expert) reads:                      │
│            • task_agent.py source code                  │
│            • failure cases from last eval               │
│            • previous patch history                     │
│          → rewrites task_agent.py                       │
│          TaskAgent (new version) → new score            │
│          Best score survives, tree grows                │
└─────────────────────────────────────────────────────────┘
```

- **Task Agent** (`python/task_agent.py`) — solves domain tasks with Chain-of-Thought reasoning
- **Meta Agent** (`python/meta_agent.py`) — studies failures and rewrites the Task Agent's code
- **Evolution Loop** — iterates for N generations, selecting the best-scoring lineage

---

## Install

```bash
git clone <repo>
cd HyperAgents-Locally
bash install.sh          # Python venv + dependencies
bash install.sh --rust   # also build the Rust binary (~30s)
bash install.sh --mlx    # also install Apple Silicon MLX support
```

Then edit `.env` to set your model and API keys (created automatically from `.env.example`).

## Quick start

Python loop:
```bash
python python/loop.py --domain factory --model ollama/llama3.2 --max-generation 8 --verbose
```

Rust loop:
```bash
cd rust
cargo run -- --domain factory --model ollama/llama3.2 --max-generation 8 --verbose
```

The Python harness also supports `search_arena` and `paper_review`; the Rust binary currently focuses on `text_classify`, `emotion`, and `factory`.

---

## Configuration (`.env`)

```bash
# ── llama.cpp server (direct GGUF, fastest local option) ─────────────
# Start the server first (once):
#   llama-server -m ~/models/gemma-4-31B-it-Q4_K_M.gguf --port 8080 -c 8192 -ngl 99
LLAMACPP_BASE_URL=http://localhost:8080
MODEL_NAME=llamacpp/gemma-4-31B-it-Q4_K_M   # label after / is cosmetic

# ── Local Ollama ──────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
# MODEL_NAME=ollama/gemma4:e4b

# ── Apple Silicon (MLX) ───────────────────────────────────────────────
# MODEL_NAME=mlx/BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit

# ── OpenRouter (300+ models, many free) ───────────────────────────────
# OPENROUTER_API_KEY=sk-or-...
# MODEL_NAME=openrouter/google/gemma-3-4b-it:free
```

### llama.cpp quick-start

```bash
# Install llama.cpp (if not already)
brew install llama.cpp          # macOS
# or build from source: https://github.com/ggml-org/llama.cpp

# Start server with your GGUF (loads once, stays hot)
llama-server \
  -m ~/models/gemma-4-31B-it-Q4_K_M.gguf \
  --port 8080 \
  -c 8192 \       # context window
  -ngl 99         # GPU layers — set to 0 for CPU-only

# Then run any loop command as normal:
source venv/bin/activate
python python/loop.py --domain factory --model llamacpp/gemma-4-31B-it-Q4_K_M

# Or with the Rust binary:
./rust/target/release/hyperagents --domain factory --model llamacpp/gemma-4-31B-it-Q4_K_M
./rust/target/release/hyperagents-comms --task relay --model llamacpp/gemma-4-31B-it-Q4_K_M
```

> **Tip:** llama.cpp is significantly faster than Ollama for large models because it skips the Ollama daemon layer. A 31B Q4 model on Apple Silicon M-series typically runs at 15–25 tok/s.

### OpenRouter free models

| Model | `MODEL_NAME` |
|---|---|
| Gemma 3 4B | `openrouter/google/gemma-3-4b-it:free` |
| Llama 4 Scout | `openrouter/meta-llama/llama-4-scout:free` |
| Qwen3 8B | `openrouter/qwen/qwen3-8b:free` |
| DeepSeek R1 | `openrouter/deepseek/deepseek-r1-0528:free` |

> **Note:** Free models are rate-limited to 20 req/min — use `--num-workers 1`.

---

## Python — Running the hyper loop

```bash
# Activate venv first
source venv/bin/activate

# Ollama (local) — factory domain (hardest, 5-class)
python python/loop.py --domain factory --model ollama/gemma4:e4b --max-generation 8 --num-workers 3 --verbose

# OpenRouter — Gemma 3 4B (free, num-workers 1)
python python/loop.py --domain factory --model openrouter/google/gemma-3-4b-it:free --max-generation 8 --num-workers 1 --verbose

# OpenRouter — Qwen3 8B (free, num-workers 1)
python python/loop.py --domain factory --model openrouter/qwen/qwen3-8b:free --max-generation 8 --num-workers 1 --verbose

# Apple Silicon MLX
python python/loop.py --domain factory --model mlx/BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit --max-generation 5
```

### All options

```
python python/loop.py [OPTIONS]

  --domain           {text_classify, emotion, rust, factory, search_arena, paper_review}
  --model            Model string (ollama/*, openrouter/*, mlx/*)
  --max-generation   Number of evolution generations  [default: 5]
  --num-samples      Samples per eval, -1 for all     [default: -1]
  --num-workers      Parallel eval threads            [default: 4]
  --parent-selection {best, latest, proportional}     [default: best]
  --output-dir       Where to write results           [default: ./outputs_local]
  --verbose / -v     Stream all subprocess output live
```

### 3-terminal workflow

**Terminal 1 — run**
```bash
source venv/bin/activate
python python/loop.py --domain factory --model openrouter/google/gemma-3-4b-it:free --max-generation 8 --num-samples 20 --num-workers 1 --verbose > /tmp/hyperloop.log 2>&1
```

**Terminal 2 — live log**
```bash
tail -f /tmp/hyperloop.log
```

**Terminal 3 — score graph**
```bash
watch -n3 'LATEST=$(ls outputs_local/ | sort | tail -1) && echo "Run: $LATEST" && cat outputs_local/$LATEST/archive.jsonl 2>/dev/null | python3 -c "import sys,json
for line in sys.stdin:
    r=json.loads(line)
    bar=\"#\"*int(r.get(\"score\",0)*30)
    print(f\"  Gen {r[\"gen\"]:>2}  {r.get(\"score\",0):.3f}  {bar}\")
" || echo "  waiting..."'
```

---

## Agent-to-Agent Communication Loop

A second mode where two AI agents exchange messages to complete a task as efficiently as possible, coached in real time by a third **Overseer** agent that scores each round and gives specific tips.

```
Round 1:  Agent A ──────────────────────────────────► Agent B
          (verbose, hedging, redundant)
                                 ◄──────────────────── Agent B
                            Overseer: score 4/10 — tips for both agents

Round 2:  Agent A ───────────────────────────────────► Agent B
          (tighter, using agreed shorthand)
                                  ◄────────────────── Agent B
                            Overseer: score 7/10 ↑ — further tips

Round N:  maximum signal / minimum words
```

### Tasks

| Task | Description |
|---|---|
| `relay` | A has a dense fact-sheet; must relay it to B in as few words as possible. Overseer quizzes B. |
| `collaborate` | Each agent has half the data. Must combine efficiently to produce a joint answer. |
| `protocol` | Repeated relay rounds with fresh data — agents develop compressed notation / shorthand over time. |
| `free` | Open discussion on a topic; overseer coaches for information density. |

### Run

```bash
source venv/bin/activate

# Relay (default) — 4 rounds, agents find efficient briefing style
python python/comms/loop.py --task relay --model ollama/llama3.2

# Collaborative puzzle — 3 rounds
python python/comms/loop.py --task collaborate --rounds 3

# Protocol compression — 5 rounds, agents develop shorthand
python python/comms/loop.py --task protocol --rounds 5

# Free discussion (random topic)
python python/comms/loop.py --task free

# Custom topic
python python/comms/loop.py --task free --topic "trade-offs in distributed systems"

# Different models for agents vs overseer
python python/comms/loop.py --task relay \
  --agent-model ollama/llama3.2 \
  --overseer-model openrouter/google/gemma-3-4b-it:free
```

### All options

```
python python/comms/loop.py [OPTIONS]

  --task              {relay, collaborate, protocol, free}  [default: relay]
  --model             Model for all three agents
  --agent-model       Override model for Agent A and B only
  --overseer-model    Override model for the Overseer only
  --rounds            Number of rounds              [default: task-specific]
  --exchanges         Exchanges per round           [default: task-specific]
  --scenario          Scenario index (0-based)      [default: 0]
  --topic             Discussion topic (free task)
  --output-dir        Output directory              [default: ./outputs_comms]
```

Output is saved to `outputs_comms/comms_<task>_<timestamp>/session.json`.

### Rust — Agent communication loop

The same communication loop is also compiled as a native Rust binary.

```bash
# Build (included in install.sh --rust)
cd rust && cargo build --release

# Relay — agents relay a project brief, improving each round
./rust/target/release/hyperagents-comms --task relay --model ollama/llama3.2

# Collaborative puzzle — 3 rounds
./rust/target/release/hyperagents-comms --task collaborate --rounds 3

# Protocol compression — 5 rounds, agents develop shorthand
./rust/target/release/hyperagents-comms --task protocol --rounds 5

# Free discussion
./rust/target/release/hyperagents-comms --task free --topic "what makes a good abstraction?"

# Split models — small agents, smarter overseer
./rust/target/release/hyperagents-comms --task relay \
  --agent-model ollama/llama3.2 \
  --overseer-model openrouter/google/gemma-3-4b-it:free
```

All options mirror the Python version (`--task`, `--model`, `--agent-model`, `--overseer-model`, `--rounds`, `--exchanges`, `--scenario`, `--topic`, `--output-dir`).

---

## Rust — Running the hyper loop

### Build

```bash
# Via install script (recommended)
bash install.sh --rust

# Or directly
cd rust && cargo build --release
# binary: rust/target/release/hyperagents
```

### Run

```bash
# Ollama (local) — factory domain
./rust/target/release/hyperagents --domain factory --model ollama/qwen2.5-coder:7b --max-generation 8 --num-workers 4 --parent-selection best --verbose

# OpenRouter — Gemma 3 4B (free, num-workers 1)
./rust/target/release/hyperagents --domain factory --model openrouter/google/gemma-3-4b-it:free --max-generation 8 --num-workers 1 --verbose

# OpenRouter — Qwen3 8B (free, num-workers 1)
./rust/target/release/hyperagents --domain factory --model openrouter/qwen/qwen3-8b:free --max-generation 8 --num-workers 1 --verbose
```

Or via cargo:

```bash
cd rust && cargo run --release -- --domain factory --model openrouter/google/gemma-3-4b-it:free --max-generation 8 --num-workers 1 --verbose
```

### All Rust options

```
hyperagents [OPTIONS]

  --domain           {text_classify, emotion, factory, search_arena, paper_review}
  --model            Model string (ollama/*, openrouter/*)  [default: ollama/llama3.2]
  --max-generation   Evolution generations                  [default: 5]
  --num-samples      Samples per eval, -1 for all          [default: -1]
  --num-workers      Rayon parallel threads                [default: 4]
  --output-dir       Output directory                       [default: ./outputs_local]
  --parent-selection {best, latest, proportional}          [default: best]
  --verbose / -v     Verbose output
```

---

## Domains

| Domain | Labels | Description | Python | Rust |
|---|---|---|:---:|:---:|
| `text_classify` | positive / negative / neutral | Sentiment classification — good baseline | ✓ | ✓ |
| `emotion` | joy / anger / fear / … | Emotion classification | ✓ | ✓ |
| `factory` | expedite / prioritize_urgent / rebalance / batch_production / optimize_throughput | Virtual factory floor dispatch — 5-class, rule-based, hardest | ✓ | ✓ |
| `rust` | compiles / borrow_error / type_error | Rust compile-error classification | ✓ | — |
| `search_arena` | a / b | Which of two search responses is better (CSV dataset required) | ✓ | — |
| `paper_review` | accept / reject / … | Academic paper outcome (CSV dataset required) | ✓ | — |

---

## Reading the results

All output lands in `outputs_local/run_YYYYMMDD_HHMMSS/`:

```
run_20260414_164911/
├── archive.jsonl          # generation scores + lineage
├── best_task_agent.py     # best evolved agent code
├── gen_initial/           # baseline evaluation
├── gen_1/
│   ├── agent_output/
│   │   ├── meta_agent_chat_history.md   # what the expert thought
│   │   └── model_patch.diff             # the code change it made
│   └── agent_evals/
│       └── chat_history_<id>.md         # per-sample reasoning
└── gen_2/ …
```

```bash
# Score summary
cat outputs_local/$(ls outputs_local/ | sort | tail -1)/archive.jsonl

# Best evolved agent
cat outputs_local/$(ls outputs_local/ | sort | tail -1)/best_task_agent.py
```

---

## Project structure

```
├── install.sh                # One-command setup
├── requirements.txt          # Python dependencies
├── .env.example              # Configuration template
├── python/                   # Python implementation
│   ├── loop.py               # Evolution loop — main entry point
│   ├── task_agent.py         # Task agent — evolves each generation
│   ├── meta_agent.py         # Meta agent — reads code + failures, writes patch
│   ├── run_meta_agent.py     # Run meta agent standalone
│   ├── agent/
│   │   ├── llm.py            # LLM interface: Ollama / MLX / OpenRouter
│   │   ├── llm_withtools.py  # Tool-use loop + fuzzy JSON parser
│   │   ├── base_agent.py     # Base agent class
│   │   └── tools/            # Editor + Bash tools
│   ├── domains/
│   │   ├── text_classify/    # Sentiment (20 train / 15 val / 15 test)
│   │   ├── emotion/          # Emotion classification
│   │   ├── factory/          # Virtual factory dispatch (5-class, hardest)
│   │   ├── rust/             # Rust compile-error classification
│   │   ├── search_arena/     # Search quality comparison (CSV)
│   │   ├── paper_review/     # Academic review outcome (CSV)
│   │   ├── harness.py        # Parallel evaluation harness
│   │   └── report.py         # Accuracy + per-label metrics
│   ├── utils/
│   │   ├── git_utils.py      # Git reset / clean / patch apply
│   │   ├── common.py         # JSON extraction helpers
│   │   └── thread_logger.py  # Thread-safe file logging
│   └── comms/
│       ├── loop.py           # Agent-to-agent comms loop — main entry point
│       ├── agents.py         # CommunicatingAgent + OverseerAgent
│       └── tasks.py          # Task definitions (relay, collaborate, protocol, free)
└── rust/                     # Rust port of the evolution loop
    ├── Cargo.toml
    └── src/
        ├── main.rs           # CLI entry point
        ├── runner.rs         # Evolution loop
        ├── llm.rs            # HTTP LLM client (Ollama + OpenRouter)
        ├── agent/            # Task + Meta agent
        ├── domains/          # Domain harness + datasets
        ├── tools/            # Editor + Bash tools
        └── utils/            # Shared utilities
```

---

## Citation & License

See the original [HyperAgents paper](https://arxiv.org/abs/2603.19461) and [LICENSE.md](LICENSE.md).
