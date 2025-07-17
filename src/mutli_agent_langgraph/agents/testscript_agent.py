from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from src.mutli_agent_langgraph.state.state import State

def testscript(state:State,retriver,user_message,model)->State:
    test_case = state.get("testcase")

    if  test_case:

        test_script_prompt = f"""
        You are a Test Script Generator Agent.

        Task:
        Write a Python Selenium test script implementing ONLY the steps and locators provided in the test case and context.

        Guardrails & Constraints:
        - DO NOT invent new steps, element locators, actions, or assertions.
        - Use step order, field names, and locators exactly as provided.
        - If an element locator or step is missing, output: "# ERROR: Step or locator missing: [describe missing piece]" at the appropriate place in the code.
        - Properly add waits, error handling, or logic that are industry standard.
        - Do NOT make assumptions about the application flow.
        - If insufficient information is provided to produce a script, output: "# ERROR: Insufficient context to generate test script."
        - Use the Test case provided and Retrived Context and create the test script

        Output Format:
        Output only valid Python code (with clear error comments if needed). Do not include any explanation or text outside the code block.

        Test Case:
        \"\"\"{test_case}\"\"\"

        Element Locators/Context:
        \"\"\"{retriver}\"\"\"

        User Query:

        \"\"\"{user_message}"\"\"
        """
    else:

        
        test_script_prompt = f"""
        You are a Test Script Generator Agent.

        Task:
        Write a Python Selenium test script implementing ONLY the steps and locators provided in the test case and context.

        Guardrails & Constraints:
        - DO NOT invent new steps, element locators, actions, or assertions.
        - Use step order, field names, and locators exactly as provided.
        - If an element locator or step is missing, output: "# ERROR: Step or locator missing: [describe missing piece]" at the appropriate place in the code.
        - Properly add waits, error handling, or logic that are industry standard.
        - Do NOT make assumptions about the application flow.
        - If insufficient information is provided to produce a script, output: "# ERROR: Insufficient context to generate test script."
        - Use the context provided for the UI flow and the locators

        Output Format:
        Output only valid Python code (with clear error comments if needed). Do not include any explanation or text outside the code block.

        Element/ UI Flow / Locators/ Context:
        \"\"\"{retriver}\"\"\"

        User Query:

        \"\"\"{user_message}"\"\"
        """
    
    context = [
        SystemMessage(content=test_script_prompt),
        HumanMessage(content=user_message)
    ]

    response = model.invoke(context)

    state["testscript"] = response.content
    state["messages"].append(AIMessage(content=response.content))
    return state