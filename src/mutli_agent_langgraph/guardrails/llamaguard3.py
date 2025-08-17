from __future__ import annotations
import subprocess, sys
from typing import Dict, Any, Literal
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import (
    log_params, log_metrics, log_json_artifact
)

Label = Literal["safe", "unsafe"]

def _safe_decode(b: bytes) -> str:
    # Try utf-8 first, fall back to 'replace' to avoid Windows cp1252 crashes
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("utf-8", errors="replace")

class LlamaGuard3:
    """Local jailbreak/safety gate using Ollama's llama-guard3 model."""
    def __init__(self, model: str = "llama-guard3") -> None:
        self.model = model

    def _run_ollama(self, text: str) -> str:
        # Use Popen and read raw bytes, then decode safely.
        proc = subprocess.Popen(
            ["ollama", "run", self.model, text or ""],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out_b, err_b = proc.communicate()
        out = _safe_decode(out_b)
        err = _safe_decode(err_b)
        if proc.returncode != 0:
            raise RuntimeError(f"Ollama command failed with code {proc.returncode}: {err.strip()}")
        else:
            print(f"Ollama output: {out.strip()}")
        # Optionally append stderr if stdout is empty
        return out if out.strip() else err

    def classify(self, text: str, direction: Literal["input","output"], session_id: str) -> Dict[str, Any]:
        try:
            raw = self._run_ollama(text)
            out = raw.strip().lower()

            # 1) Prefer explicit "unsafe" over "safe" to avoid the substring trap
            # 2) Use word boundaries when possible
            import re
            if re.search(r"\bunsafe\b", out):
                label: Label = "unsafe"
            elif re.search(r"\bsafe\b", out):
                label = "safe"
            else:
                # Optional: try to parse JSON if the model returns structured output
                # (Many guard models output { "verdict": "unsafe" } or similar)
                try:
                    import json
                    j = json.loads(raw)
                    verdict = str(j.get("verdict", "")).lower()
                    if verdict == "unsafe":
                        label = "unsafe"
                    elif verdict == "safe":
                        label = "safe"
                    else:
                        label = "safe"  # default fail-open (flip if you prefer fail-closed)
                except Exception:
                    label = "safe"
        except Exception as e:
            raw = f"llamaguard_error:{e}"
            label = "safe"  # flip to "unsafe" if you prefer fail-closed

        log_params({"moderation_model": self.model})
        log_metrics({f"moderation_{direction}_safe": 1.0 if label == "safe" else 0.0})
        log_json_artifact({
            "direction": direction,
            "label": label,
            "raw": raw
        }, f"moderation_{direction}_{session_id}.json")
        return {"direction": direction, "label": label, "raw": raw}


    def enforce_or_refuse(self, text: str, direction: Literal["input","output"], session_id: str) -> Dict[str, Any]:
        res = self.classify(text, direction, session_id)
        if res["label"] == "unsafe":
            return {"allowed": False, "reason": f"llamaguard_{direction}_unsafe", "message": "I can’t assist with that request."}
        return {"allowed": True, "reason": "llamaguard_allow"}
