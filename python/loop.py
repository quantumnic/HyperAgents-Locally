import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime

# ── Path anchors ──────────────────────────────────────────────────────────────
# PYTHON_DIR: the python/ subdirectory (where all Python source lives)
# PROJECT_ROOT: repo root (where .git, rust/, README are)
PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PYTHON_DIR)

# Make sure agent/, domains/, utils/ are importable from python/
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from utils.common import file_exist_and_not_empty

# Global verbose flag
VERBOSE = False


def run_command(cmd, workdir=None, timeout=3600, stream=False):
    """Run a command as subprocess with timeout. Returns (exit_code, stdout, stderr).

    If stream=True (or VERBOSE is set), streams output in real-time instead of capturing.
    """
    cmd_str = shlex.join(cmd) if isinstance(cmd, list) else cmd
    print(f"  [CMD] {cmd_str}")
    should_stream = stream or VERBOSE
    try:
        if should_stream:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, shell=isinstance(cmd, str), bufsize=1,
            )
            stdout_lines = []
            start_time = time.time()
            for line in process.stdout:
                line = line.rstrip('\n')
                stdout_lines.append(line)
                print(f"  │ {line}")
                if timeout and (time.time() - start_time) > timeout:
                    process.kill()
                    print(f"  [TIMEOUT] after {timeout}s")
                    return -1, '\n'.join(stdout_lines), "timeout"
            process.wait()
            return process.returncode, '\n'.join(stdout_lines), ""
        else:
            result = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True,
                timeout=timeout, shell=isinstance(cmd, str),
            )
            if result.stdout.strip():
                print(f"  [OUT] {result.stdout.strip()[:500]}")
            if result.returncode != 0 and result.stderr.strip():
                print(f"  [ERR] {result.stderr.strip()[:500]}")
            return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] after {timeout}s")
        return -1, "", "timeout"
    except Exception as e:
        print(f"  [ERROR] {e}")
        return -1, "", str(e)


def git_diff(workdir, base_commit):
    """Get git diff against base commit."""
    code, stdout, _ = run_command(["git", "diff", base_commit], workdir=workdir)
    return stdout if code == 0 else ""


def git_reset(workdir, commit):
    """Reset git to a commit."""
    run_command(["git", "reset", "--hard", commit], workdir=workdir)
    run_command(["git", "clean", "-fd"], workdir=workdir)


def git_apply_diff(workdir, diff_file):
    """Apply a diff file."""
    if os.path.exists(diff_file) and os.path.getsize(diff_file) > 0:
        code, _, _ = run_command(["git", "apply", "--allow-empty", diff_file], workdir=workdir)
        return code == 0
    return False


def format_archive_value(value, precision=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def get_base_commit(workdir):
    """Get current HEAD commit."""
    code, stdout, _ = run_command(["git", "rev-parse", "HEAD"], workdir=workdir)
    return stdout.strip() if code == 0 else "HEAD"


def get_score_from_report(report_path):
    """Extract accuracy score from a report.json."""
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
        return report.get("overall_accuracy", None)
    except Exception:
        return None


def run_initial_eval(project_dir, domain, model, output_dir, num_samples=-1, subset="_train", num_workers=4):
    """Run initial evaluation of the base task agent."""
    print(f"\n{'='*60}")
    print(f"Running initial evaluation on {domain} (Workers: {num_workers})...")
    print(f"{'='*60}")
    start_time = time.time()

    run_id = f"initial_{domain}{subset}_0"
    cmd = [
        sys.executable, "-m", "domains.harness",
        "--agent_path", os.path.join(PYTHON_DIR, "task_agent.py"),
        "--output_dir", output_dir,
        "--run_id", run_id,
        "--domain", domain,
        "--num_samples", str(num_samples),
        "--num_workers", str(num_workers),
        "--subset", subset,
    ]
    run_command(cmd, workdir=PYTHON_DIR, timeout=1800, stream=VERBOSE)

    # Generate report
    eval_dir = os.path.join(output_dir, run_id)
    cmd = [
        sys.executable, "-m", "domains.report",
        "--domain", domain,
        "--dname", eval_dir,
    ]
    run_command(cmd, workdir=PYTHON_DIR, timeout=300)

    elapsed = time.time() - start_time
    score = get_score_from_report(os.path.join(eval_dir, "report.json"))
    print(f"  Initial score: {score} ({elapsed:.1f}s)")
    return score


def run_meta_agent(project_dir, model, output_dir, base_commit, evals_folder, iterations_left=5):
    """Run the meta agent to produce modifications."""
    print(f"\n  Running meta agent (model={model})...")
    start_time = time.time()

    agent_output_dir = os.path.join(output_dir, "agent_output")
    os.makedirs(agent_output_dir, exist_ok=True)
    chat_history_file = os.path.join(agent_output_dir, "meta_agent_chat_history.md")

    cmd = [
        sys.executable, os.path.join(PYTHON_DIR, "run_meta_agent.py"),
        "--model", model,
        "--chat_history_file", chat_history_file,
        "--repo_path", PYTHON_DIR + "/",
        "--evals_folder", evals_folder,
        "--git_dir", PROJECT_ROOT,
        "--base_commit", base_commit,
        "--outdir", agent_output_dir,
        "--iterations_left", str(iterations_left),
    ]
    # Always stream meta agent output so you can see what it's doing
    code, _, _ = run_command(cmd, workdir=PYTHON_DIR, timeout=3600, stream=True)

    elapsed = time.time() - start_time
    patch_file = os.path.join(agent_output_dir, "model_patch.diff")
    success = code == 0 and file_exist_and_not_empty(patch_file)

    # Show patch summary in verbose mode
    status = "SUCCEEDED" if success else "FAILED"
    print(f"  Meta agent {status} ({elapsed:.1f}s)")
    if success and VERBOSE:
        try:
            with open(patch_file) as f:
                patch_content = f.read()
            lines_changed = len([line for line in patch_content.splitlines() if line.startswith('+') or line.startswith('-')])

            print(f"  Patch: {lines_changed} lines changed")
            print("  ┌─── patch preview ───")
            for line in patch_content.splitlines()[:30]:
                print(f"  │ {line}")
            if patch_content.count('\n') > 30:
                print(f"  │ ... ({patch_content.count(chr(10)) - 30} more lines)")
            print("  └─────────────────────")
        except Exception:
            pass

    return success, patch_file


def run_eval(project_dir, domain, model, output_dir, gen_id, num_samples=-1, subset="_train", num_workers=4):
    """Evaluate the current task agent."""
    print(f"  Evaluating generation {gen_id} (Workers: {num_workers})...")
    start_time = time.time()

    run_id = f"{domain}_eval"
    eval_output_dir = os.path.join(output_dir, run_id)
    os.makedirs(eval_output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "domains.harness",
        "--agent_path", os.path.join(PYTHON_DIR, "task_agent.py"),
        "--output_dir", output_dir,
        "--run_id", run_id,
        "--domain", domain,
        "--num_samples", str(num_samples),
        "--num_workers", str(num_workers),
        "--subset", subset,
    ]
    run_command(cmd, workdir=PYTHON_DIR, timeout=1800, stream=VERBOSE)

    # Generate report
    cmd = [
        sys.executable, "-m", "domains.report",
        "--domain", domain,
        "--dname", eval_output_dir,
    ]
    run_command(cmd, workdir=PYTHON_DIR, timeout=300)

    elapsed = time.time() - start_time
    score = get_score_from_report(os.path.join(eval_output_dir, "report.json"))
    print(f"  Score for gen {gen_id}: {score} ({elapsed:.1f}s)")
    return score


def select_parent(archive, method="best"):
    """Simple parent selection from archive."""
    if not archive:
        return "initial"

    valid = [a for a in archive if a.get("score") is not None]
    if not valid:
        return archive[-1]["id"]

    if method == "best":
        return max(valid, key=lambda x: x["score"])["id"]
    elif method == "latest":
        return valid[-1]["id"]
    else:  # random/proportional
        import random
        scores = [max(a["score"], 0.01) for a in valid]
        total = sum(scores)
        probs = [s / total for s in scores]
        return random.choices(valid, weights=probs, k=1)[0]["id"]


def print_evolution_tree(archive):
    """Print an ASCII representation of the evolution tree."""
    if not archive:
        return

    # Build adjacency list
    adj = {}
    nodes = {str(a["id"]): a for a in archive}
    for a in archive:
        pid = str(a["parent"]) if a["parent"] is not None else None
        if pid not in adj:
            adj[pid] = []
        adj[pid].append(str(a["id"]))

    print("\n🌱 Evolution Tree:")

    def print_node(node_id, prefix="", is_last=True):
        entry = nodes.get(node_id)
        if not entry:
            return

        score = format_archive_value(entry.get("score"))
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}Gen {node_id} (Score: {score})")

        child_prefix = prefix + ("    " if is_last else "│   ")
        children = adj.get(node_id, [])
        for i, child_id in enumerate(children):
            print_node(child_id, child_prefix, i == len(children) - 1)

    # Start from root (parent is None)
    roots = adj.get(None, [])
    for i, root_id in enumerate(roots):
        print_node(root_id, "", i == len(roots) - 1)


def generate_loop_local(
    domain="text_classify",
    model=None,
    max_generation=5,
    num_samples=-1,
    output_dir_parent=None,
    parent_selection="best",
    verbose=False,
    num_workers=4,
):
    """Main local generation loop — no Docker required."""
    global VERBOSE
    VERBOSE = verbose

    if model is None:
        from agent.llm import DEFAULT_MODEL
        model = DEFAULT_MODEL

    # Set the model in env so all subprocesses pick it up
    os.environ["MODEL_NAME"] = model

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir_parent is None:
        output_dir_parent = os.path.join(PROJECT_ROOT, "outputs_local")
    output_dir = os.path.join(output_dir_parent, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    project_dir = PROJECT_ROOT   # git repo root (for git ops)
    base_commit = get_base_commit(project_dir)
    archive = []
    archive_file = os.path.join(output_dir, "archive.jsonl")

    print(f"{'='*60}")
    print("HyperAgents Local Loop")
    print(f"  Model:      {model}")
    print(f"  Domain:     {domain}")
    print(f"  Gens:       {max_generation}")
    print(f"  Output:     {output_dir}")
    print(f"  Base commit:{base_commit[:8]}")
    print(f"{'='*60}")

    # --- Initial evaluation ---
    subset = "_train" if domain in ("text_classify", "rust", "factory") else ""
    initial_score = run_initial_eval(
        project_dir, domain, model, os.path.join(output_dir, "gen_initial"),
        num_samples=num_samples, subset=subset, num_workers=num_workers,
    )
    archive.append({"id": "initial", "parent": None, "score": initial_score, "gen": 0})
    with open(archive_file, "a") as f:
        f.write(json.dumps(archive[-1]) + "\n")

    # --- Generation loop ---
    loop_start = time.time()
    for gen_id in range(1, max_generation + 1):
        print(f"\n{'='*60}")
        print(f"Generation {gen_id}/{max_generation}")
        # Show score history so far
        scores = [f"{('initial' if a['id'] == 'initial' else 'gen_' + str(a['id']))}: {a['score']}" for a in archive if a.get('score') is not None]
        if scores:
            print(f"  Score history: {' → '.join(scores)}")
        best_so_far = max((a.get('score', 0) or 0 for a in archive), default=0)
        print(f"  Best so far:  {best_so_far}")
        print(f"{'='*60}")

        parent_id = select_parent(archive, method=parent_selection)
        parent_score = next((a.get('score') for a in archive if a['id'] == parent_id), None)
        print(f"  Parent: {parent_id} (score: {parent_score})")

        gen_output_dir = os.path.join(output_dir, f"gen_{gen_id}")
        os.makedirs(gen_output_dir, exist_ok=True)

        # Reset to base
        git_reset(project_dir, base_commit)

        # Apply parent diffs (lineage)
        parent_entry = next((a for a in archive if a["id"] == parent_id), None)
        if parent_entry and parent_entry.get("patch_file"):
            parent_patch_applied = git_apply_diff(project_dir, parent_entry["patch_file"])
            if not parent_patch_applied:
                print(f"  Parent patch failed to apply, skipping generation {gen_id}.")
                success = False
                patch_file = None
                score = None
                evals_folder = output_dir
                entry = {
                    "id": gen_id,
                    "parent": parent_id,
                    "score": score,
                    "gen": gen_id,
                    "patch_file": None,
                    "meta_success": False,
                }
                archive.append(entry)
                with open(archive_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                continue

        # Run meta agent
        evals_folder = output_dir  # meta agent can see previous gen results
        success, patch_file = run_meta_agent(
            project_dir, model, gen_output_dir, base_commit,
            evals_folder, iterations_left=max_generation - gen_id,
        )

        score = None
        if success:
            # Apply the new diff and evaluate
            git_reset(project_dir, base_commit)
            # Re-apply parent lineage
            if parent_entry and "patch_file" in parent_entry:
                parent_patch_applied = git_apply_diff(project_dir, parent_entry["patch_file"])
                if not parent_patch_applied:
                    print("  Parent patch failed to re-apply after reset, skipping evaluation.")
                    success = False
            # Apply new modifications
            patch_applied = git_apply_diff(project_dir, patch_file) if success else False
            if not patch_applied:
                print("  Generated patch failed to apply cleanly, skipping evaluation.")
                success = False

            if success:
                score = run_eval(
                    project_dir, domain, model, gen_output_dir,
                    gen_id, num_samples=num_samples, subset=subset,
                    num_workers=num_workers,
                )
        else:
            print("  Meta agent failed, skipping evaluation.")

        # Store in archive
        entry = {
            "id": gen_id,
            "parent": parent_id,
            "score": score,
            "gen": gen_id,
            "patch_file": patch_file if success else None,
            "meta_success": success,
        }
        archive.append(entry)
        with open(archive_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Save metadata
        metadata = {
            "gen_id": gen_id,
            "parent_id": parent_id,
            "score": score,
            "model": model,
            "domain": domain,
            "meta_success": success,
        }
        with open(os.path.join(gen_output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Show improvement indicator
        if score is not None and parent_score is not None:
            delta = score - parent_score
            indicator = f"{'📈' if delta > 0 else '📉' if delta < 0 else '➡️'} {delta:+.4f}"
        else:
            indicator = "N/A"
        print(f"  Gen {gen_id} done — score: {score}, parent: {parent_id}, change: {indicator}")

    # --- Final reset ---
    git_reset(project_dir, base_commit)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for entry in archive:
        marker = " ⭐" if entry.get("score") == max(
            (a.get("score", 0) or 0 for a in archive)
        ) else ""
        gen_text = format_archive_value(entry.get("id"), precision=0)
        score_text = format_archive_value(entry.get("score"))
        parent_text = "-" if entry.get("parent") is None else str(entry.get("parent"))
        print(f"  Gen {gen_text:>8} | Score: {score_text:>8} | Parent: {parent_text:>8}{marker}")

    print_evolution_tree(archive)

    best = max((a for a in archive if a.get("score") is not None), key=lambda x: x["score"], default=None)
    if best:
        print(f"\n  Best: Gen {best['id']} with score {best['score']:.3f}")

        # Export the best agent's source code for convenience
        git_reset(project_dir, base_commit)
        if best["id"] != "initial" and best.get("patch_file"):
            git_apply_diff(project_dir, best["patch_file"])

        best_agent_path = os.path.join(output_dir, "best_task_agent.py")
        shutil.copy(os.path.join(PYTHON_DIR, "task_agent.py"), best_agent_path)
        print(f"  Best agent source exported to: {best_agent_path}")

        # Final cleanup reset
        git_reset(project_dir, base_commit)

    total_elapsed = time.time() - loop_start
    print(f"\n  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Output saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperAgents Local Loop (Docker-free)")
    parser.add_argument("--domain", type=str, default="text_classify",
                        choices=["text_classify", "search_arena", "paper_review", "rust", "factory"],
                        help="Domain to optimize on")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to use (e.g., ollama/llama3.2, mlx/BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit)")
    parser.add_argument("--max-generation", type=int, default=5,
                        help="Number of evolution generations")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel evaluation workers")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--parent-selection", type=str, default="best",
                        choices=["best", "latest", "proportional"],
                        help="Parent selection method")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output: stream all subprocess output, show patch previews and detailed progress")
    args = parser.parse_args()

    generate_loop_local(
        domain=args.domain,
        model=args.model,
        max_generation=args.max_generation,
        num_samples=args.num_samples,
        output_dir_parent=args.output_dir,
        parent_selection=args.parent_selection,
        verbose=args.verbose,
        num_workers=args.num_workers,
    )
