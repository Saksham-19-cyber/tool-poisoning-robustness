import json
import os
from typing import Any, Dict, List

def load_tasks(task_path: str = None) -> List[Dict[str, Any]]:
    if task_path is None:
        base_dir = os.path.dirname(__file__)
        task_path = os.path.join(base_dir, "tasks.json")
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_task_by_id(task_id: str, task_path: str = None) -> Dict[str, Any]:
    tasks = load_tasks(task_path)
    for t in tasks:
        if t.get("id") == task_id:
            return t
    raise KeyError(f"Task with id '{task_id}' not found.")
