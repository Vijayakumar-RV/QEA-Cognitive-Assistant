# utils/tc_clean.py
import json, re, ast
from typing import Any, Dict, List

# ---------- parsing ----------
def parse_json_loose(text_or_obj: Any):
    if isinstance(text_or_obj, (list, dict)):
        return text_or_obj
    s = str(text_or_obj).strip()
    # remove code fences if the LLM added them
    s = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    # strict JSON
    try: return json.loads(s)
    except: pass
    # first {...} or [...]
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if m:
        chunk = m.group(1)
        for loader in (json.loads, ast.literal_eval):
            try: return loader(chunk)
            except: pass
    return None

# ---------- text cleaners ----------
NOTE_PAREN = re.compile(r"\s*\(Note:[^)]+\)", flags=re.IGNORECASE)  # (Note: ... )
INFER_BRACK = re.compile(r"\s*\[[^\]]*inferred[^\]]*\]", flags=re.IGNORECASE)  # [Inferred ...]

def clean_text(s: str) -> str:
    if not isinstance(s, str): return s
    s = NOTE_PAREN.sub("", s)
    s = INFER_BRACK.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ensure_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def clean_list(items: List[str]) -> List[str]:
    out, seen = [], set()
    for it in _ensure_list(items):
        t = clean_text(str(it))
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ---------- normalize to canonical schema ----------
KEEP_KEYS = {
    "Test Case ID","Test Case Title","Description","Preconditions",
    "Test Steps","Expected Result","Test Type","Priority","Tags"
}

def normalize_test_cases(obj,testcase_format) -> List[Dict]:
    cases = obj if isinstance(obj, list) else obj.get("items", [])
    norm = []
    for tc in cases:
        if not isinstance(tc, dict): 
            continue
        pre  = clean_list(tc.get("Preconditions", []))
        if testcase_format.lower() == "gherkin":
            step = [tc.get("Test Steps", []).replace("\n", "<br>")]
        else:
            step = clean_list(tc.get("Test Steps", []))
        tags = clean_list(tc.get("Tags", []))
        norm.append({
            "Test Case ID": clean_text(tc.get("Test Case ID","")),
            "Test Case Title": clean_text(tc.get("Test Case Title","")),
            "Description": clean_text(tc.get("Description","")),
            "Preconditions": pre,
            "Test Steps": step,
            "Expected Result": clean_text(tc.get("Expected Result","")),
            "Test Type": clean_text(tc.get("Test Type","")),
            "Priority": clean_text(tc.get("Priority","")),
            "Tags": tags,
        })
    return norm

# ---------- validation (optional but useful) ----------
def validate_cases(cases: List[Dict]) -> List[str]:
    errs = []
    for i, tc in enumerate(cases):
        missing = [k for k in KEEP_KEYS if k not in tc or tc[k] == "" or tc[k] == []]
        if missing:
            errs.append(f"Case {i} ({tc.get('Test Case ID','')}) missing: {', '.join(missing)}")
    return errs

# ---------- flat rows for CSV ----------
def cases_to_rows(cases: List[Dict]) -> List[Dict]:
    rows = []
    for tc in cases:
        rows.append({
            "Test Case ID": tc["Test Case ID"],
            "Title": tc["Test Case Title"],
            "Type": tc["Test Type"],
            "Priority": tc["Priority"],
            "Tags": ", ".join(tc["Tags"]),
            "Preconditions": " | ".join(tc["Preconditions"]),
            "Test Steps": "\n".join(tc["Test Steps"]),
            "Expected Result": tc["Expected Result"],
            "Description": tc["Description"],
        })
    return rows

# ---------- pretty markdown for UI ----------
def cases_to_markdown(cases: List[Dict]) -> str:
    parts = []
    for tc in cases:
        parts.append(
f"""### {tc['Test Case ID']} — {tc['Test Case Title']}
**Type:** {tc['Test Type']}  |  **Priority:** {tc['Priority']}  
**Tags:** {", ".join(tc['Tags'])}

**Description**  
{tc['Description']}

**Preconditions**
""" + "\n".join(f"- {p}" for p in tc["Preconditions"]) + "\n\n" +
"**Test Steps**\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(tc["Test Steps"])) + "\n\n" +
f"**Expected Result**\n> {tc['Expected Result']}\n"
        )
    return "\n---\n".join(parts)

# ---------- compact memory string (to feed next LLM turn) ----------
def cases_to_memory(cases: List[Dict], max_cases=3, max_steps=3) -> str:
    if not cases: return ""
    lines = ["Previous test cases (summary):"]
    for tc in cases[:max_cases]:
        lines.append(f"- {tc['Test Case ID']}: {tc['Test Case Title']} (Priority={tc['Priority']})")
        for i, s in enumerate(tc["Test Steps"][:max_steps], 1):
            lines.append(f"  {i}. {s}")
        if len(tc["Test Steps"]) > max_steps:
            lines.append(f"  ... (+{len(tc['Test Steps'])-max_steps} more)")
        lines.append(f"  Expected: {tc['Expected Result']}")
    return "\n".join(lines)


import re
from typing import Dict, Any, List

GHERKIN_KWS = r"(Feature:|Background:|Scenario(?: Outline)?:|Examples:|Given|When|Then|And|But|\|)"

def normalize_gherkin_multiline(text: str) -> str:
    # Ensure each keyword starts on its own line (but keep the first if already at start)
    # Insert newline before any keyword that is not at the start of a line.
    text = re.sub(rf"\s*(?<!\n)({GHERKIN_KWS})\s+", r"\n\1 ", text)

    # Collapse accidental double spaces after keywords
    text = re.sub(rf"({GHERKIN_KWS})\s+", r"\1 ", text)

    # Remove any leftover spaces at line starts
    text = re.sub(r"\n\s+", "\n", text)

    # Strip outer whitespace
    return text.strip()

def repair_gherkin_in_json(cases: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode.lower() != "gherkin":
        return cases
    for obj in cases:
        steps = obj.get("Test Steps")
        print(f"Original Test Steps for {obj.get('Test Case ID', '')}: {steps}")
        if isinstance(steps, str):
            fixed = normalize_gherkin_multiline(steps)
            obj["Test Steps"] = fixed
    return cases
