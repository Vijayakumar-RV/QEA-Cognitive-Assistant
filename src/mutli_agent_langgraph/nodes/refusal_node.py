from __future__ import annotations
from typing import Dict, Any
from src.mutli_agent_langgraph.state.state import State
from langchain_core.messages import AIMessage
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import log_metrics

REFUSAL_TEXT = "I can’t help with that request."

def refusal_node(state: State) -> State:
    safety = state.get("safety", {})
    msg = REFUSAL_TEXT
    if safety.get("blocked_by") == "llamaguard_input":
        msg = "I can’t assist with that request."
    elif safety.get("should_refuse", False):
        msg = "I don’t have enough verified project context to answer. Please sync the flow or upload the page schema."

    state["messages"].append(AIMessage(content=msg))
    log_metrics({"refusals": 1.0})
    return state
