"""Build EvoTool data from all 500 Retail train tasks plus 20 dev tasks."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join(REPO, "data_taubench_official", "taubench")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taubench_tools.json")

sys.path.insert(0, REPO)
from src.envs.taubench_env import taubench_success  # noqa: E402


class Action:
    def __init__(self, name=None, kwargs=None, **extra):
        self.name = name
        self.kwargs = kwargs if kwargs is not None else {}


class Task:
    def __init__(self, annotator=None, user_id=None, instruction=None,
                 actions=None, outputs=None, **extra):
        self.user_id = user_id
        self.instruction = instruction
        self.actions = actions or []
        self.outputs = list(outputs or [])


def load_tasks(retail: str, filename: str, list_var: str) -> list[Task]:
    path = os.path.join(retail, filename)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    source = re.sub(r"^from tau_bench\.types import .*$", "", source, flags=re.MULTILINE)
    namespace = {"Task": Task, "Action": Action}
    exec(compile(source, path, "exec"), namespace)
    return namespace[list_var]


def convert(task: Task, split: str, index: int, tools: list[dict]) -> dict:
    return {
        "id": f"tau_retail_{split}_{index}",
        "query": task.instruction or "",
        "available_tools": tools,
        "gold_plan": [{"tool": action.name, "args": action.kwargs} for action in task.actions],
        "gold_answer": None,
        "mock_outputs": {},
        "data_split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-repo", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    retail = os.path.join(args.tau_repo, "tau_bench", "envs", "retail")
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        tools = json.load(f)

    sources = [
        ("train", "tasks_train.py", "TASKS_TRAIN", 500),
        ("dev", "tasks_dev.py", "TASKS_DEV", 20),
    ]
    samples = []
    required_outputs = {}
    for split, filename, variable, expected in sources:
        tasks = load_tasks(retail, filename, variable)
        if len(tasks) != expected:
            raise RuntimeError(f"expected {expected} {split} tasks, found {len(tasks)}")
        for index, task in enumerate(tasks, start=1):
            item = convert(task, split, index, tools)
            if not taubench_success(item, item["gold_plan"]):
                raise RuntimeError(f"gold plan failed for {item['id']}")
            samples.append(item)
            required_outputs[item["id"]] = task.outputs

    os.makedirs(args.out_dir, exist_ok=True)
    samples_path = os.path.join(args.out_dir, "samples.json")
    outputs_path = os.path.join(args.out_dir, "required_outputs.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(required_outputs, f, indent=2, ensure_ascii=False)
    print(f"wrote train=500 dev=20 -> {samples_path}")
    print(f"wrote required outputs -> {outputs_path}")


if __name__ == "__main__":
    main()
