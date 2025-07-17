from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from src.mutli_agent_langgraph.state.state import State

def userstory(retriver,user_message,model,state:State)->State:

    user_story_prompt =f""" You are a User Story Agent for a software testing assistant.

            **Task:**  
            Given the user request and the following context, retrieve and present the relevant user story as clearly as possible.  
            - Use only the information provided in the context—do NOT invent or hallucinate user stories, requirements, IDs, or acceptance criteria.
            - If the exact user story (by ID or description) is found in the context, present it using this format:

            User Story ID: <ID>
            Title: <Title>
            Description: <Description>
            Acceptance Criteria:
            - <Criterion 1>
            - <Criterion 2>
            ...

            - If the user story is not found or the context does not include a match, reply exactly with:  
            "No matching user story found for the request."

            - Do not summarize or rephrase unless explicitly instructed.  
            - Do not generate new user stories or IDs.
            
            ** Retrived User Stories:{retriver} **
            **User Request:** {user_message}


        """

    context = [
        SystemMessage(content=user_story_prompt),
        HumanMessage(content=user_message)
    ]

    response = model.invoke(context)

    state["user"] = response.content
    state["messages"].append(AIMessage(content=response.content))
    return state