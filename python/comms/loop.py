"""
Agent-to-Agent Communication Loop
==================================
Two agents exchange messages to complete a task as efficiently as possible.
A third Overseer agent watches every round, scores efficiency, and gives tips.

Usage:
    python python/comms/loop.py --task relay --rounds 4 --model ollama/llama3.2
    python python/comms/loop.py --task collaborate --rounds 3
    python python/comms/loop.py --task protocol --rounds 5
    python python/comms/loop.py --task free --topic "distributed systems trade-offs"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import random
import textwrap
from datetime import datetime

_PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_PYTHON_DIR)
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from agent.llm import DEFAULT_MODEL
from comms.agents import CommunicatingAgent, OverseerAgent
from comms.tasks import TASKS


# ── ANSI colours ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
BLUE   = "\033[94m"

def _c(text, *codes):
    return "".join(codes) + str(text) + RESET

def _wrap(text: str, width: int = 88, indent: str = "    ") -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width, subsequent_indent=indent))
    return "\n".join(lines)

def _banner(title: str, char: str = "═") -> str:
    width = 72
    pad = max(0, (width - len(title) - 2) // 2)
    return _c(char * pad + f" {title} " + char * (width - pad - len(title) - 2), BOLD)

def _section(title: str) -> str:
    return _c(f"── {title} " + "─" * max(0, 60 - len(title)), DIM)


# ── output helpers ────────────────────────────────────────────────────────────

def _print_message(agent_name: str, other_name: str, text: str, color) -> None:
    arrow = f"{_c(agent_name, BOLD, color)} → {_c(other_name, DIM)}"
    words = len(text.split())
    header = f"{arrow}  {_c(f'[{words}w]', DIM)}"
    print(f"\n{header}")
    print(_c(_wrap(text), color))


def _print_overseer(feedback: dict, round_num: int, scores: list[int]) -> None:
    score = feedback["score"]
    score_color = GREEN if score >= 7 else (YELLOW if score >= 4 else RED)
    bar = "█" * score + "░" * (10 - score)

    print(f"\n{_section('OVERSEER')}")
    print(f"  Score: {_c(score, BOLD, score_color)}/10  {_c(bar, score_color)}")
    if feedback["verdict"]:
        print(f"  {_c(feedback['verdict'], DIM)}")

    if feedback["tips_a"]:
        print(f"\n  {_c('Tips for Agent A:', BOLD, CYAN)}")
        for tip in feedback["tips_a"]:
            print(f"    {_c('•', CYAN)} {_wrap(tip, indent='      ')}")

    if feedback["tips_b"]:
        print(f"\n  {_c('Tips for Agent B:', BOLD, MAGENTA)}")
        for tip in feedback["tips_b"]:
            print(f"    {_c('•', MAGENTA)} {_wrap(tip, indent='      ')}")

    if len(scores) > 1:
        trend = "↑" if scores[-1] > scores[-2] else ("↓" if scores[-1] < scores[-2] else "→")
        trend_color = GREEN if trend == "↑" else (RED if trend == "↓" else DIM)
        history = "  ".join(
            f"{_c(s, GREEN if s >= 7 else (YELLOW if s >= 4 else RED))}"
            for s in scores
        )
        print(f"\n  Score history: {history}  {_c(trend, BOLD, trend_color)}")


# ── task runners ──────────────────────────────────────────────────────────────

def run_relay(
    agent_a: CommunicatingAgent,
    agent_b: CommunicatingAgent,
    overseer: OverseerAgent,
    scenario: dict,
    rounds: int,
    exchanges: int,
    output_log: list,
) -> None:
    print(_c(f"\nTask: RELAY  |  Scenario: {scenario['id']}", BOLD))
    print(_c(f"Goal: relay key facts from A to B, B answers quiz questions.", DIM))
    print()

    for rnd in range(1, rounds + 1):
        print(_banner(f"Round {rnd} / {rounds}"))

        a_system = (
            f"You are Agent A in a communication efficiency experiment.\n\n"
            f"YOUR TASK:\n{scenario['a_briefing']}\n\n"
            f"Relay the above information to Agent B as efficiently as possible. "
            f"Every word costs you — eliminate filler, hedging, and repetition. "
            f"Use shorthand, bullet points, or any compression you and B agree on."
        )
        b_system = (
            f"You are Agent B in a communication efficiency experiment.\n\n"
            f"YOUR ROLE:\n{scenario['b_briefing']}\n\n"
            f"Receive information from Agent A. Ask focused clarifying questions "
            f"only if something is genuinely missing. Acknowledge receipt concisely."
        )

        agent_a.start_round(a_system)
        agent_b.start_round(b_system)

        exchange_log = []

        # A opens the conversation
        opening = (
            f"Agent B, I have a briefing to relay. Ready? "
            f"Round {rnd} — applying efficiency improvements."
            if rnd > 1
            else "Agent B, incoming briefing:"
        )

        msg = opening
        for ex in range(exchanges):
            # A → B
            reply_b = agent_b.receive(msg)
            _print_message("Agent A", "Agent B", msg, CYAN)
            if ex == 0:
                # first "message" is A's opening; now A sends the real briefing
                relay_msg = f"[round {rnd}] " + scenario["a_briefing"].strip()
                exchange_log.append({"agent": "A", "text": relay_msg})
                reply_b2 = agent_b.receive(relay_msg)
                _print_message("Agent A (brief)", "Agent B", relay_msg, CYAN)
                _print_message("Agent B", "Agent A", reply_b2, MAGENTA)
                exchange_log.append({"agent": "B", "text": reply_b2})
                msg = reply_b2
                continue

            exchange_log.append({"agent": "A", "text": msg})
            _print_message("Agent B", "Agent A", reply_b, MAGENTA)
            exchange_log.append({"agent": "B", "text": reply_b})

            # B → A follow-up (A clarifies if needed)
            if ex < exchanges - 1:
                reply_a = agent_a.receive(reply_b)
                _print_message("Agent A", "Agent B", reply_a, CYAN)
                exchange_log.append({"agent": "A", "text": reply_a})
                msg = reply_a

        feedback = overseer.evaluate(
            exchange_log=exchange_log,
            task_context=f"Relay scenario: {scenario['id']}",
            verification_questions=scenario.get("quiz_questions"),
            agent_b_last_reply=exchange_log[-1]["text"] if exchange_log else "",
        )
        _print_overseer(feedback, rnd, overseer.scores)

        agent_a.add_tip(f"Round {rnd}: " + "; ".join(feedback["tips_a"]))
        agent_b.add_tip(f"Round {rnd}: " + "; ".join(feedback["tips_b"]))

        output_log.append({
            "round": rnd,
            "exchange": exchange_log,
            "score": feedback["score"],
            "verdict": feedback["verdict"],
            "tips_a": feedback["tips_a"],
            "tips_b": feedback["tips_b"],
            "avg_words_a": round(agent_a.avg_words, 1),
            "avg_words_b": round(agent_b.avg_words, 1),
        })
        print()


def run_collaborate(
    agent_a: CommunicatingAgent,
    agent_b: CommunicatingAgent,
    overseer: OverseerAgent,
    scenario: dict,
    rounds: int,
    exchanges: int,
    output_log: list,
) -> None:
    print(_c(f"\nTask: COLLABORATE  |  Scenario: {scenario['id']}", BOLD))
    print(_c(f"Goal: {scenario['joint_goal']}", DIM))
    print()

    for rnd in range(1, rounds + 1):
        print(_banner(f"Round {rnd} / {rounds}"))

        a_system = (
            f"You are Agent A in a collaborative information-sharing experiment.\n\n"
            f"YOUR PRIVATE INFORMATION:\n{scenario['a_briefing']}\n\n"
            f"Share your information with Agent B efficiently. Together you must: "
            f"{scenario['joint_goal']}\n"
            f"Be concise — every redundant word wastes a turn. "
            f"Work toward a joint answer the overseer can verify."
        )
        b_system = (
            f"You are Agent B in a collaborative information-sharing experiment.\n\n"
            f"YOUR PRIVATE INFORMATION:\n{scenario['b_briefing']}\n\n"
            f"Share your information with Agent A efficiently. Together you must: "
            f"{scenario['joint_goal']}\n"
            f"Be concise — every redundant word wastes a turn. "
            f"Once you have enough information, produce the joint answer."
        )

        agent_a.start_round(a_system)
        agent_b.start_round(b_system)

        exchange_log = []
        msg = "Let's share our information efficiently and compute the joint answer."

        for ex in range(exchanges):
            reply_a = agent_a.receive(msg) if ex > 0 else scenario["a_briefing"].split("\n")[0].strip()
            if ex == 0:
                # A starts by sharing
                share_a = agent_a.receive(msg)
                _print_message("Agent A", "Agent B", share_a, CYAN)
                exchange_log.append({"agent": "A", "text": share_a})
                share_b = agent_b.receive(share_a)
                _print_message("Agent B", "Agent A", share_b, MAGENTA)
                exchange_log.append({"agent": "B", "text": share_b})
                msg = share_b
                continue

            _print_message("Agent A", "Agent B", reply_a, CYAN)
            exchange_log.append({"agent": "A", "text": reply_a})
            reply_b = agent_b.receive(reply_a)
            _print_message("Agent B", "Agent A", reply_b, MAGENTA)
            exchange_log.append({"agent": "B", "text": reply_b})
            msg = reply_b

        feedback = overseer.evaluate(
            exchange_log=exchange_log,
            task_context=f"Collaborative scenario: {scenario['id']}\nJoint goal: {scenario['joint_goal']}",
            verification_questions=[scenario["joint_goal"]],
            agent_b_last_reply=exchange_log[-1]["text"] if exchange_log else "",
        )
        _print_overseer(feedback, rnd, overseer.scores)

        agent_a.add_tip(f"Round {rnd}: " + "; ".join(feedback["tips_a"]))
        agent_b.add_tip(f"Round {rnd}: " + "; ".join(feedback["tips_b"]))

        output_log.append({
            "round": rnd,
            "exchange": exchange_log,
            "score": feedback["score"],
            "verdict": feedback["verdict"],
        })
        print()


def run_protocol(
    agent_a: CommunicatingAgent,
    agent_b: CommunicatingAgent,
    overseer: OverseerAgent,
    task_def: dict,
    rounds: int,
    output_log: list,
) -> None:
    print(_c("\nTask: PROTOCOL — develop compressed notation across rounds", BOLD))
    print(_c("Goal: relay structured data in fewer and fewer words each round.", DIM))
    print()

    data_rounds = task_def["data_rounds"]
    developed_protocol = []  # shared shorthand agreed upon so far

    for rnd in range(1, rounds + 1):
        data = data_rounds[(rnd - 1) % len(data_rounds)]
        print(_banner(f"Round {rnd} / {rounds}  [{data['project']}]"))

        proto_note = ""
        if developed_protocol:
            proto_note = (
                f"\n\nPROTOCOL DEVELOPED SO FAR (use and extend this):\n"
                + "\n".join(f"  {p}" for p in developed_protocol)
            )

        a_system = task_def["a_system"] + proto_note
        b_system = task_def["b_system"] + proto_note

        agent_a.start_round(a_system)
        agent_b.start_round(b_system)

        # Serialize the data to relay
        data_str = json.dumps(data, indent=2)
        a_prompt = (
            f"New data to relay (round {rnd}):\n{data_str}\n\n"
            f"Send this to Agent B as compactly as possible."
        )

        exchange_log = []
        msg_a = agent_a.receive(a_prompt)
        _print_message("Agent A", "Agent B", msg_a, CYAN)
        exchange_log.append({"agent": "A", "text": msg_a})

        # B receives and reconstructs
        b_reply = agent_b.receive(
            msg_a + "\n\n[Reconstruct the full report from Agent A's message above.]"
        )
        _print_message("Agent B", "Agent A", b_reply, MAGENTA)
        exchange_log.append({"agent": "B", "text": b_reply})

        # Verification questions
        vqs = [
            f"What is the project name?",
            f"What is the status?",
            f"What is the completion percentage?",
            f"Who is the owner?",
            f"What is the ETA in days?",
        ]

        feedback = overseer.evaluate(
            exchange_log=exchange_log,
            task_context=f"Protocol compression task. Original data:\n{data_str}",
            verification_questions=vqs,
            agent_b_last_reply=b_reply,
        )
        _print_overseer(feedback, rnd, overseer.scores)

        # Extract any protocol suggestions from overseer tips
        for tip in feedback["tips_a"] + feedback["tips_b"]:
            if any(kw in tip.lower() for kw in ["shorthand", "abbreviat", "schema", "notation", "code", "prefix", "symbol"]):
                developed_protocol.append(tip)

        agent_a.add_tip(f"Round {rnd}: " + "; ".join(feedback["tips_a"]))
        agent_b.add_tip(f"Round {rnd}: " + "; ".join(feedback["tips_b"]))

        wc_a = sum(len(e["text"].split()) for e in exchange_log if e["agent"] == "A")
        print(f"\n  {_c('Words used by A this round:', DIM)} {wc_a}")

        output_log.append({
            "round": rnd,
            "data": data,
            "exchange": exchange_log,
            "score": feedback["score"],
            "words_a": wc_a,
        })
        print()


def run_free(
    agent_a: CommunicatingAgent,
    agent_b: CommunicatingAgent,
    overseer: OverseerAgent,
    topic: str,
    rounds: int,
    exchanges: int,
    output_log: list,
) -> None:
    print(_c(f"\nTask: FREE DISCUSSION", BOLD))
    print(_c(f"Topic: {topic}", DIM))
    print()

    for rnd in range(1, rounds + 1):
        print(_banner(f"Round {rnd} / {rounds}"))

        base_instruction = (
            f"You are one of two AI agents having an efficient discussion about:\n"
            f'"{topic}"\n\n'
            f"Your goal: exchange maximum insight in minimum words. "
            f"Cut hedging ('I think', 'it seems', 'perhaps'), cut filler, "
            f"cut repetition of what the other agent already said. "
            f"Make every sentence carry new information or a new argument."
        )

        agent_a.start_round("You are Agent A. " + base_instruction)
        agent_b.start_round("You are Agent B. " + base_instruction)

        exchange_log = []
        msg = f"Topic: {topic}"

        for ex in range(exchanges):
            reply_a = agent_a.receive(msg)
            _print_message("Agent A", "Agent B", reply_a, CYAN)
            exchange_log.append({"agent": "A", "text": reply_a})

            reply_b = agent_b.receive(reply_a)
            _print_message("Agent B", "Agent A", reply_b, MAGENTA)
            exchange_log.append({"agent": "B", "text": reply_b})
            msg = reply_b

        feedback = overseer.evaluate(
            exchange_log=exchange_log,
            task_context=f"Free discussion topic: {topic}",
        )
        _print_overseer(feedback, rnd, overseer.scores)

        agent_a.add_tip(f"Round {rnd}: " + "; ".join(feedback["tips_a"]))
        agent_b.add_tip(f"Round {rnd}: " + "; ".join(feedback["tips_b"]))

        output_log.append({
            "round": rnd,
            "exchange": exchange_log,
            "score": feedback["score"],
            "verdict": feedback["verdict"],
        })
        print()


# ── final summary ─────────────────────────────────────────────────────────────

def _final_summary(scores: list[int], output_log: list, out_path: str) -> None:
    print(_banner("FINAL SUMMARY"))
    if scores:
        avg = sum(scores) / len(scores)
        best = max(scores)
        trend_total = scores[-1] - scores[0] if len(scores) > 1 else 0
        trend_str = (
            _c(f"+{trend_total} improvement!", BOLD, GREEN)
            if trend_total > 0
            else (_c(f"{trend_total} (no gain)", BOLD, RED) if trend_total < 0 else _c("→ steady", DIM))
        )
        print(f"\n  Rounds: {len(scores)}")
        print(f"  Scores: {' → '.join(str(s) for s in scores)}")
        print(f"  Avg score:   {avg:.1f}/10")
        print(f"  Best round:  {best}/10")
        print(f"  Trend: {trend_str}")
        print()
        # ASCII score chart
        print("  Score chart:")
        for i, s in enumerate(scores):
            bar = "█" * s + "░" * (10 - s)
            color = GREEN if s >= 7 else (YELLOW if s >= 4 else RED)
            print(f"    Round {i+1:>2}  {_c(bar, color)}  {s}/10")

    print(f"\n  Output saved to: {_c(out_path, BOLD)}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent-to-Agent Communication Loop with Overseer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python python/comms/loop.py --task relay --model ollama/llama3.2
          python python/comms/loop.py --task collaborate --rounds 3
          python python/comms/loop.py --task protocol --rounds 5
          python python/comms/loop.py --task free --topic "trade-offs in distributed systems"
          python python/comms/loop.py --task relay --agent-model ollama/llama3.2 --overseer-model openrouter/google/gemma-3-4b-it:free
        """),
    )
    parser.add_argument(
        "--task", default="relay",
        choices=["relay", "collaborate", "protocol", "free"],
        help="Communication task type (default: relay)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model for all three agents (A, B, overseer). Override individually below.",
    )
    parser.add_argument("--agent-model", default=None, help="Model for Agent A and B")
    parser.add_argument("--overseer-model", default=None, help="Model for the Overseer")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--exchanges", type=int, default=None, help="Exchanges per round")
    parser.add_argument(
        "--scenario", type=int, default=0,
        help="Scenario index (0-based) for relay/collaborate tasks",
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Discussion topic for --task free",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for output JSON")
    args = parser.parse_args()

    agent_model    = args.agent_model    or args.model
    overseer_model = args.overseer_model or args.model

    task_def = TASKS[args.task]
    rounds   = args.rounds    or task_def.get("rounds_default", 4)
    exchanges= args.exchanges or task_def.get("exchanges_per_round", 3)

    # ── header ──
    print(_banner("AGENT COMMS — Self-Optimising Communication Loop"))
    print(f"  Task:           {_c(args.task.upper(), BOLD)}")
    print(f"  Agent model:    {_c(agent_model, BOLD)}")
    print(f"  Overseer model: {_c(overseer_model, BOLD)}")
    print(f"  Rounds:         {rounds}")
    print(f"  Exchanges/rnd:  {exchanges}")
    print()
    print(_c(f"  {task_def['description']}", DIM))
    print()

    agent_a  = CommunicatingAgent("A", agent_model)
    agent_b  = CommunicatingAgent("B", agent_model)
    overseer = OverseerAgent(overseer_model)
    output_log: list = []

    # ── run ──
    try:
        if args.task == "relay":
            scenarios = task_def["scenarios"]
            scenario  = scenarios[args.scenario % len(scenarios)]
            run_relay(agent_a, agent_b, overseer, scenario, rounds, exchanges, output_log)

        elif args.task == "collaborate":
            scenarios = task_def["scenarios"]
            scenario  = scenarios[args.scenario % len(scenarios)]
            run_collaborate(agent_a, agent_b, overseer, scenario, rounds, exchanges, output_log)

        elif args.task == "protocol":
            run_protocol(agent_a, agent_b, overseer, task_def, rounds, output_log)

        elif args.task == "free":
            topic = args.topic
            if not topic:
                topics = task_def["topics"]
                topic  = topics[random.randint(0, len(topics) - 1)]
                print(_c(f"  (random topic selected)", DIM))
            run_free(agent_a, agent_b, overseer, topic, rounds, exchanges, output_log)
    except KeyboardInterrupt:
        print(_c("\nInterrupted by user — saving the partial session.", YELLOW))

    # ── save ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "outputs_comms")
    run_dir = os.path.join(out_dir, f"comms_{args.task}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "session.json")
    with open(out_path, "w") as f:
        json.dump({
            "task": args.task,
            "agent_model": agent_model,
            "overseer_model": overseer_model,
            "rounds": rounds,
            "scores": overseer.scores,
            "log": output_log,
        }, f, indent=2)

    _final_summary(overseer.scores, output_log, out_path)


if __name__ == "__main__":
    main()
