# utils/session_store.py
from pathlib import Path
import json

BASE = Path("artifacts/session_store"); BASE.mkdir(parents=True, exist_ok=True)

def save_session_rows(session_id: str, rows):
    p = BASE / f"{session_id}_test_rows.json"
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)

def load_session_rows(session_id: str):
    p = BASE / f"{session_id}_test_rows.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []

def save_session_script_blob(session_id: str, code: str):
    p = BASE / f"{session_id}_test_script_blob.txt"
    p.write_text(code or "", encoding="utf-8")
    return str(p)

def load_session_script_blob(session_id: str) -> str:
    p = BASE / f"{session_id}_test_script_blob.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


# def save_session_scripts_dict(session_id: str, scripts: Dict[str, str]):
#     p = BASE / f"{session_id}_test_scripts.json"
#     # merge with existing so we don't lose previously generated scripts
#     existing = {}
#     if p.exists():
#         try:
#             existing = json.loads(p.read_text(encoding="utf-8"))
#         except Exception:
#             existing = {}
#     existing.update(scripts or {})
#     p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
#     return str(p)

# def load_session_scripts_dict(session_id: str) -> Dict[str, str]:
#     p = BASE / f"{session_id}_test_scripts.json"
#     if p.exists():
#         return json.loads(p.read_text(encoding="utf-8"))
#     return {}