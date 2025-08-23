from __future__ import annotations
import json, os, re
from typing import Any, Dict, List
from src.mutli_agent_langgraph.state.state import State
import json, os, re
from pathlib import Path

try:
    import mlflow
except Exception:
    mlflow = None

try:
    from openai import AzureOpenAI
except Exception as e:
    AzureOpenAI = None

JUDGE_DEPLOYMENT = os.getenv("JUDGE_DEPLOYMENT", "gpt-5-mini")  # Azure *deployment name*
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT","https://2023a-mc6p2p97-eastus2.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY","9E5kngq3RBt7s0N6LIW9ol0XFDP1jNBYoB0FevNQXlj6jeZ3XnjBJQQJ99BFACHYHv6XJ3w3AAAAACOGXjfO")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")


ROOT = Path(__file__).resolve().parents[2]  # points to src/mutli_agent_langgraph/
PROFILE_PATH = ROOT / "resources" / "judge_profile.md"
RUBRICS_DIR  = ROOT / "resources" / "rubrics"
print(f"Judge profile path: {PROFILE_PATH}")
BASE_RUBRIC = """
You are a strict evaluator for Quality Engineering & Assurance outputs.
Return JSON only:
{
  "task_score": float (0.0-10.0),
  "completeness": float (0.0-1.0),
  "correctness": float (0.0-1.0),
  "step_coverage": float (0.0-1.0),
  "formatting": float (0.0-1.0),
  "reasoning": float (0.0-1.0),
  "harmful": boolean,
  "comments": string
}
Scoring must follow the project profile and the (optional) per-usecase rubric exactly.
If any critical requirement is missing or contradicted by contexts, reduce scores sharply.
""".strip()

def _safe_parse_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {
            "task_score": 0.0, "completeness": 0.0, "correctness": 0.0,
            "step_coverage": 0.0, "formatting": 0.0, "reasoning": 0.0,
            "harmful": False, "comments": "Judge JSON parse failed."
        }

def _rule_cov(ans: str, must: List[str]):
    if not must: return True, 1.0
    a = (ans or "").lower()
    hits = sum(1 for m in must if m.lower() in a)
    return hits == len(must), hits / max(1, len(must))

def _read_text(p: Path, default: str = "") -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return default

def _usecase_rubric(usecase: str) -> str:
    # map usecase → file name; adjust if your usecase keys differ
    key = (usecase or "").strip().lower()
    fname = None
    if key == "qea_assistant":
        fname = "qea_assistant.md"
    elif key in ("qea document assistant", "document assistant"):
        fname = "document_assistant.md"
    # add more mappings as needed
    if fname and (RUBRICS_DIR / fname).exists():
        print(f"Using rubric for {key}: {fname}")
        print(f"Rubric path: {RUBRICS_DIR / fname}")
        return _read_text(RUBRICS_DIR / fname)
    return ""


def _mlflow_active_run_id() -> str | None:
    if mlflow is None:
        return None
    try:
        r = mlflow.active_run()
        return r.info.run_id if r else None
    except Exception:
        return None

def _mlflow_log_safely(evaluation: Dict[str, Any],
                       payload: Dict[str, Any],
                       system_prompt: str,
                       usecase: str,
                       session_id: str):
    """Log to the current MLflow run if active; no-ops if MLflow isn't available."""
    run_id = _mlflow_active_run_id()
    if not run_id:
        return
    try:
        # 1) flat metrics (only simple numerics)
        j = evaluation.get("judge", {}) if evaluation else {}
        metrics = {
            "judge.task_score": float(j.get("task_score", 0.0)),
            "judge.completeness": float(j.get("completeness", 0.0)),
            "judge.correctness": float(j.get("correctness", 0.0)),
            "judge.step_coverage": float(j.get("step_coverage", 0.0)),
            "judge.formatting": float(j.get("formatting", 0.0)),
            "judge.reasoning": float(j.get("reasoning", 0.0)),
            "judge.harmful": 1.0 if j.get("harmful") else 0.0,
            "judge.rule_pass": 1.0 if evaluation.get("rule_pass") else 0.0,
            "judge.rule_coverage_pct": float(evaluation.get("rule_coverage_pct", 0.0)),
        }
        mlflow.log_metrics(metrics)

        # 2) params/tags (diagnostics)
        mlflow.set_tags({
            "judge.usecase": usecase or "",
            "judge.session_id": session_id or "",
            "judge.deployment": JUDGE_DEPLOYMENT,
            "judge.api_version": AZURE_OPENAI_API_VERSION,
        })

        # 3) artifacts (JSON payloads)
        artifact = {
            "evaluation": evaluation,
            "payload": {**payload, "contexts": (payload.get("contexts") or [])[:5]},  # limit artifacts size
            "system_prompt_head": system_prompt[:4000],
        }
        # Prefer mlflow.log_dict; fallback to log_text if unavailable
        try:
            mlflow.log_dict(artifact, "judge/judge_evaluation.json")
        except Exception:
            mlflow.log_text(json.dumps(artifact, ensure_ascii=False, indent=2), "judge/judge_evaluation.json")
    except Exception:
        # Never break the graph because of MLflow logging
        pass



def create_judge_node():
    if AzureOpenAI is None:
        raise RuntimeError("pip install openai>=1.40.0")

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    project_profile = _read_text(PROFILE_PATH, default="# (missing) judge_profile.md")
    # keep around 10–15k chars budget to avoid overlong system prompts
    project_profile = project_profile[:15000]

    def judge_node(state: State) -> State:
        # 1) Gather prompt, answer, contexts
        user_prompt, model_answer = "", ""
        msgs = state.get("messages", [])
        for m in msgs:
            if getattr(m, "type", None) == "human" or m.__class__.__name__ == "HumanMessage":
                user_prompt = getattr(m, "content", "") or user_prompt
        if msgs:
            last = msgs[-1]
            model_answer = getattr(last, "content", "") if last else ""

        contexts = state.get("retrived_content", []) or []
        must = (state.get("eval_gold") or {}).get("must_include", [])
        usecase = "qea_assistant"
        session_id = str(state.get("session_id", ""))

        # 2) Rule coverage (simple signal)
        rule_pass, cov = _rule_cov(model_answer, must)

        # 3) Compose system prompt = base rubric + project profile + per-usecase rubric
        per_usecase = _usecase_rubric(usecase)
        system_prompt = (
            BASE_RUBRIC
            + "\n\n--- PROJECT PROFILE ---\n"
            + project_profile
            + ("\n\n--- USECASE RUBRIC ---\n" + per_usecase if per_usecase else "")
        )

        payload = {
            "prompt": user_prompt,
            "model_answer": model_answer,
            "contexts": contexts,
        }

        resp = client.responses.create(
        model=JUDGE_DEPLOYMENT,      # Azure DEPLOYMENT NAME
        instructions=system_prompt,  # <-- system content goes here
        input=json.dumps(payload, ensure_ascii=False),  # <-- user content as plain string
        
        )

        # Extract output text robustly across SDK versions
        raw = getattr(resp, "output_text", None)
        if not raw:
            try:
                raw = resp.output[0].content[0].text
            except Exception:
                raw = ""
        raw = (raw or "").strip()
        data = _safe_parse_json(raw)
    
        print(f"data from the judge {data}")
        evaluation = {
            "rule_pass": bool(rule_pass),
            "rule_coverage_pct": round(cov * 100, 2),
            "judge": data
        }
        state["evaluation"] = evaluation
        _mlflow_log_safely(evaluation=evaluation,
                           payload=payload,
                           system_prompt=system_prompt,
                           usecase=usecase,
                           session_id=session_id)


        return state

    return judge_node
