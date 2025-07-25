from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from src.mutli_agent_langgraph.state.state import State

def testcase(retriver,user_message,model,state:State)->State:
    test_case_prompt = f"""
            You are a Test Case Generator Agent for software quality engineering.

            Task:
            Generate a clear, detailed, and step-by-step manual test case for the scenario request by the user, using ONLY the provided context (UI flow, user story, or requirements).

            Guardrails & Constraints:
            - Use only the UI elements and labels exactly as described in the context.
            - If an expected step, element, or navigation is clearly described, include it in the test case.
            - If a step or element is not explicitly mentioned but is implied by surrounding context (e.g., navigation between connected pages), you may cautiously infer it and mention this as: "[Inferred from context]".
            - Do not invent UI components or actions that are not present in the given context.
            - If there is truly no way to proceed, say: “Insufficient context to generate a complete test case.” Otherwise, provide the best possible partial steps and flag missing information.

            Output Format:
            Test Case Title: <Title>
            Preconditions:
            - <Precondition 1>
            Steps:
            1. <Step 1>
            2. <Step 2>
            ...
            Expected Results:
            - <Expected Result 1>
            - <Expected Result 2>

            user query:
            \"\"\"{user_message}\"\"\"

            Provided Context:
            \"\"\"{retriver}\"\"\"
            """
    context = [
            SystemMessage(content=test_case_prompt),
            HumanMessage(content=user_message)
        ]

    response = model.invoke(context)

    state["testcase"] = response.content
    state["messages"].append(AIMessage(content=response.content))
    
    return state