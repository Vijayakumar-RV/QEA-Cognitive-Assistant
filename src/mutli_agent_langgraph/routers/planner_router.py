from __future__ import annotations
from typing import Dict, Any
from src.mutli_agent_langgraph.state.state import State
def _requires_context_from_plan(plan_obj) -> bool:
    try:
        gen = getattr(plan_obj, "generation", None)
        if not gen:
            return False
        wanted = set(x.lower() for x in gen)
        return bool(wanted.intersection({"testcase", "testscript", "userstory"}))
    except Exception:
        return False

def route_after_planner(state: State) -> str:
    """Return 'to_retriver' if the plan needs RAG context, else 'to_qeacognitive'."""
    plan_obj = state.get("plan", None)
    need_ctx = _requires_context_from_plan(plan_obj)
   
    return "to_retriver" if need_ctx else "to_qeacognitive"
