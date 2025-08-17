from src.mutli_agent_langgraph.state.state import State
import datetime
from pathlib import Path
import os
import pandas as pd
import subprocess
import re
def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _first_heading(md: str) -> str:
    """
    Extract first '### <id> — <title>' line; fallback to 'tc'.
    """
    if not md:
        return "tc"
    m = re.search(r"^###\s+([^\n]+)", md, flags=re.MULTILINE)
    if not m:
        return "tc"
    # sanitize filename chunk
    chunk = m.group(1).split("—")[0].strip()  # take the id before em dash
    chunk = re.sub(r"[^A-Za-z0-9_\-]", "_", chunk)
    return chunk or "tc"


def save_testcase_excel_csv(state:State,transform):
    test_string = "### Testcase US-02-Testing"

    prefix = _first_heading(test_string)[:10]
    filename = f"{prefix}_{_timestamp()}"
    folder_path = os.getcwd()
    print(f" Test rows inside save_test excel : {state['test_rows']}")
    df = pd.DataFrame(state["test_rows"])
    if transform == "excel":
        df.to_excel(f"{folder_path}/generated_testcases/{filename}.xlsx", index=False)
    elif transform == "csv":
        df.to_csv(f"{folder_path}/generated_testcases/{filename}.csv", index=False)
    else:
        pass
    return filename


def save_testscript_execute(status, state: State):
    cmd = []
    test_script = state.get("test_script")
    test_case = state.get("test_case")
    filename = test_case.split("\n")[0].replace("### ", "").split(" — ")[0][:5] + "_test_cases"
    filename = filename.replace("-", "_")
    filename = f"{filename}_{_timestamp()}"
    folder_path = os.getcwd()

    if status == "both":
        
        with open(f"{folder_path}/generated_testscripts/{filename}.py", "w") as f:
            f.write(test_script)
        cmd = ["python", f"{folder_path}/generated_testscripts/{filename}.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        if status == "save":
            with open(f"{folder_path}/generated_testscripts/{filename}.py", "w") as f:
                f.write(test_script)
        elif status == "execute":
            cmd = ["python", f"{folder_path}/generated_testscripts/{filename}.py"]
            result = subprocess.run(cmd, capture_output=True, text=True)
        else:
            print("Invalid status provided. Use 'save', 'execute', or 'both'.")
        
    cmd.clear()
    print("Test script executed. Output:")
    print(result.stdout)
    print("Test script executed. Errors:")
    print(result.stderr)
    return state
