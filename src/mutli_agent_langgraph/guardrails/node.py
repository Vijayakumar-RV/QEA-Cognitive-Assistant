from __future__ import annotations
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.guardrails.llamaguard3 import LlamaGuard3
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import log_metrics

def create_guardrails_node():
    moderator = LlamaGuard3()
    print("Came inside the Guardrails")
    def guardrails_node(state: State) -> State:
        session_id = state["session_id"]
        messages: List[BaseMessage] = state["messages"]
        print(f"Guardrails session_id: {session_id}, messages: {messages}")
        last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        user_text = last_human.content if last_human else ""
        print(f"Guardrails user_text: {user_text}")
        # INPUT moderation first
        mod = moderator.enforce_or_refuse(user_text, "input", session_id)
        state.setdefault("safety", {})
        state["safety"]["moderation_input"] = mod
        print(f"Guardrails moderation result: {mod}")

        if not mod["allowed"]:
            # Add refusal so END still returns a reply
            state["messages"].append(AIMessage(content="I can’t assist with that request."))
            state["safety"].update({
                "blocked_by": "llamaguard_input",
                "reason": mod["reason"],
            })
            log_metrics({"guardrails_blocked": 1.0})
            return state

        # Allowed → proceed
        state["safety"].update({
            "blocked_by": None
        })
        log_metrics({"guardrails_blocked": 0.0})
        return state

    return guardrails_node

# Router after guardrails: either END (unsafe) or go to PLANNER
def route_after_guardrails(state: State) -> str:
    safety = state.get("safety", {}) or {}
    print(f"Guardrails routing with safety: {safety}")
    return "to_end" if safety.get("blocked_by") == "llamaguard_input" else "to_planner"
