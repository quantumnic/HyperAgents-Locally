import os
import sys

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd


def get_dataset(domain, subset=""):
    """Load dataset for a given domain and subset string."""
    if domain == "text_classify":
        from domains.text_classify.dataset import get_split
        split = next((s for s in ("train", "val", "test") if s in subset), "train")
        return pd.DataFrame(get_split(split))
    if domain == "rust":
        from domains.rust.dataset import get_split
        split = next((s for s in ("train", "val", "test") if s in subset), "train")
        return pd.DataFrame(get_split(split))
    if domain == "emotion":
        from domains.emotion.dataset import get_split
        split = next((s for s in ("train", "val", "test") if s in subset), "train")
        return pd.DataFrame(get_split(split))
    if domain == "factory":
        from domains.factory.dataset import get_split
        split = next((s for s in ("train", "val", "test") if s in subset), "train")
        return pd.DataFrame(get_split(split))
    if domain == "formwerk":
        from domains.formwerk.dataset import DATASET
        # Split: first 14 = train, last 6 = validation
        if "val" in subset or "test" in subset:
            data = [d for d in DATASET if d["id"].startswith("v")]
        else:
            data = [d for d in DATASET if not d["id"].startswith("v")]
        return pd.DataFrame(data)
    # CSV-based domains
    return pd.read_csv(f"./domains/{domain}/dataset{subset}.csv", dtype=str)


def run_agent(TaskAgent, model, row, evals_folder, format_input_dict, question_id_col):
    question_id = row[question_id_col]
    chat_history_path = os.path.join(evals_folder, f"chat_history_{question_id}.md")
    agent = TaskAgent(model=model, chat_history_file=chat_history_path)
    inputs = format_input_dict(row)
    prediction, _ = agent.forward(inputs)
    return prediction


def load_task_agent(agent_path: str):
    """Load a TaskAgent class from a file path or importable module path."""
    if agent_path.endswith(".py") or os.path.exists(agent_path):
        abs_path = os.path.abspath(agent_path)
        spec = importlib.util.spec_from_file_location("agent_module", abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from file: {abs_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "TaskAgent"):
            raise AttributeError(f"No TaskAgent found in file: {abs_path}")
        return mod.TaskAgent
    mod = importlib.import_module(agent_path)
    if not hasattr(mod, "TaskAgent"):
        raise AttributeError(f"No TaskAgent found in module: {agent_path}")
    return mod.TaskAgent


def harness(
    agent_path="./task_agent.py",
    output_dir="./outputs",
    run_id=None,
    domain="text_classify",
    num_samples=-1,
    save_interval=100,
    num_workers=5,
    resume_from=None,
    subset="",
):
    utils_module = importlib.import_module(f"domains.{domain}.utils")
    format_input_dict = utils_module.format_input_dict
    question_id_col = utils_module.QUESTION_ID
    model = utils_module.MODEL

    TaskAgent = load_task_agent(agent_path)

    if resume_from:
        output_folder = os.path.abspath(resume_from)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if run_id is None else run_id
        output_folder = os.path.join(os.getcwd(), output_dir, run_id)

    evals_folder = os.path.join(output_folder, "agent_evals")
    os.makedirs(evals_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "predictions.csv")

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, dtype=str)
        completed_ids = set(existing_df[~existing_df["prediction"].isna()][question_id_col])
    else:
        existing_df = None
        completed_ids = set()

    dataset = get_dataset(domain=domain, subset=subset)
    if num_samples > 0:
        dataset = dataset[:num_samples]

    if existing_df is not None:
        dataset = dataset.merge(
            existing_df[[question_id_col, "prediction"]], on=question_id_col, how="left"
        )
    else:
        dataset["prediction"] = None

    predictions = dataset["prediction"].tolist()
    if save_interval <= 0:
        save_interval = max(len(dataset), 1)
    futures = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, row in dataset.iterrows():
            if pd.notna(row["prediction"]) or row[question_id_col] in completed_ids:
                continue
            futures.append((
                i,
                executor.submit(
                    run_agent, TaskAgent, model, row, evals_folder,
                    format_input_dict, question_id_col,
                ),
            ))

        for idx, future in futures:
            try:
                predictions[idx] = future.result()
            except Exception as exc:
                question_id = dataset.iloc[idx][question_id_col]
                print(f"Warning: task {question_id} failed: {exc}")
                predictions[idx] = None
            if (idx + 1) % save_interval == 0:
                dataset["prediction"] = predictions
                dataset.to_csv(output_path, index=False)

    dataset["prediction"] = predictions
    dataset.to_csv(output_path, index=False)
    print(f"Final predictions saved to {output_path}")
    return output_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation harness for a domain.")
    parser.add_argument("--agent_path", type=str, default="./task_agent.py")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--domain", type=str, required=True,
        choices=["text_classify", "emotion", "rust", "factory", "search_arena", "paper_review"],
    )
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--subset", type=str, default="")
    args = parser.parse_args()

    harness(
        agent_path=args.agent_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
        domain=args.domain,
        num_samples=args.num_samples,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
        subset=args.subset,
    )
