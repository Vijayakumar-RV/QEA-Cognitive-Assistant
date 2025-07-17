from src.mutli_agent_langgraph.tools.retrive_knowledge import Retrive
from src.mutli_agent_langgraph.state.state import State

def retriever_agent(state:State)->dict:

    retriver = Retrive()

    plan = state.get("plan")
    user_message = state["messages"][-1].content

    if not plan:
        state["retrived_content"] = None
        return state

    retrieved_items = [item.type for item in plan.reterivals]
    try:
        if "both" in retrieved_items or {"UI_Flow","User_Story"}.issubset(set(retrieved_items)):
            
            state["retrived_content"] = retriver.clean_flow_text(user_message)+"\n\n"+retriver.clean_story_text(user_message)
            
        elif "User_Story" in retrieved_items:
            state["retrived_content"] = retriver.clean_story_text(user_message)
            
        elif "UI_Flow" in retrieved_items:
            text_flow = retriver.clean_flow_text(user_message)
            state["retrived_content"] = text_flow
        else:
            state["retrived_content"] = "No proper content retrieved"
    except Exception as e:
        print(f"Error in json {e}")

    finally:
        return state