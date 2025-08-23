from src.mutli_agent_langgraph.tools.retrive_knowledge import Retrive
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import log_json_artifact, log_metrics

def retriever_agent(state: State):
    retriver = Retrive()
    plan = state.get("plan")
    user_message = state["messages"][-1].content

    if not plan:
        state["retrived_content"] = None
        state.setdefault("safety", {})
        state["safety"]["have_context"] = False
        state["safety"]["should_refuse"] = False
        return state

    retrieved_items = [item.type for item in plan.reterivals]  # keep your field spelling

    content = ""
    try:
        if "both" in retrieved_items or {"UI_Flow", "User_Story"}.issubset(set(retrieved_items)):
            content = ((retriver.clean_flow_text(user_message) or "") + "\n\n" + (retriver.clean_story_text(user_message) or "")).strip()
        elif "User_Story" in retrieved_items:
            content = (retriver.clean_story_text(user_message) or "").strip()
        elif "UI_Flow" in retrieved_items:
            content = (retriver.clean_flow_text(user_message) or "").strip()
        else:
            content = ""
        print(f"retrived content: {content}")

        state["retrived_content"] = content if content else None

        have_ctx = bool(content)
        state.setdefault("safety", {})
        state["safety"]["have_context"] = have_ctx
        state["safety"]["should_refuse"] = not have_ctx

        log_metrics({"retrieved_context_len": (len(content))})
        log_json_artifact({
            "query": user_message,
            "have_context": have_ctx,
            "preview": content[:1200]
        }, "rag/retriever_preview.json")


    except Exception as e:
        print(f"Error in retriever_agent: {e}")
    finally:
        return state
