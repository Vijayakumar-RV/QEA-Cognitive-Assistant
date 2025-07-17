from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from src.mutli_agent_langgraph.state.state import State

def testcase(retriver,user_message,model,state:State)->State:
    test_case_prompt = f"""
            You are a Test Case Generator Agent for software quality engineering.

            Task:
            Generate a clear, detailed, and step-by-step manual test case for the scenario request by the user, using ONLY the provided context (UI flow, user story, or requirements).

            Guardrails & Constraints:
            - DO NOT invent or add any steps, UI elements, preconditions, or expected results that are not present in the context.
            - Use element names and labels exactly as given. If an element or step is missing, clearly state: "Step or element missing: [describe what is missing]".
            - Do NOT infer or assume steps, flows, or validation logic not present in the data.
            - If context is insufficient to produce a valid test case, reply with: "Insufficient context to generate a test case."

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