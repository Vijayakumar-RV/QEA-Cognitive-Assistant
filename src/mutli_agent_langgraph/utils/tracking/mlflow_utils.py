# src/multi_agent_langgraph/utils/tracking/mlflow_utils.py
import os
import time
import json
from contextlib import contextmanager
from typing import Dict, Any, Optional

import mlflow


def init_mlflow(tracking_uri: str, experiment_name: str):
    """
    Configure MLflow tracking URI and experiment.
    For local dev, use ./mlruns in your repo root.
    """
    if tracking_uri.startswith("./"):
        os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def run_mlflow_run(
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    nested: bool = False,
):
    """
    Start an MLflow run around a user action or session.
    Usage:
        with run_mlflow_run("QEA_Assistant_session", tags={...}, params={...}):
            ... your code ...
    """
    active = mlflow.active_run()
    use_nested = nested or (active is not None)
    with mlflow.start_run(run_name=run_name, nested=use_nested):
        if tags:
            mlflow.set_tags(tags)
        if params:
            # stringify to avoid param type issues
            mlflow.log_params({k: str(v) for k, v in params.items()})
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            mlflow.log_metric("run_duration_sec", duration)


def log_params(d: Dict[str, Any]) -> None:
    if not d:
        return
    mlflow.log_params({k: str(v) for k, v in d.items()})


def log_metrics(d: Dict[str, float]) -> None:
    if not d:
        return
    mlflow.log_metrics(d)


def _tmp_path_for(path: str) -> str:
    safe = path.replace("/", "_")
    base = os.path.join("/tmp", f"_mlflow_{safe}")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    return base


def log_text_artifact(text: str, path: str) -> None:
    """
    Save a small text as an MLflow artifact.
    `path` can include folders, e.g. "prompts/last_prompt.txt".
    """
    tmp = _tmp_path_for(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text or "")
    artifact_dir = os.path.dirname(path) or None
    mlflow.log_artifact(tmp, artifact_path=artifact_dir)


def log_json_artifact(obj: Any, path: str) -> None:
    """
    Save a small JSON object as an MLflow artifact.
    """
    txt = json.dumps(obj, indent=2, ensure_ascii=False)
    log_text_artifact(txt, path)


