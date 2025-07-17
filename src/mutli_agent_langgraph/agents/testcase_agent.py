from src.mutli_agent_langgraph.state.state import State
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

def testing_Agent(state:State,model)->State:
    
    plan = state["plan"].generation
    retriver = state['retrived_content']
    user_message = state["messages"][-1].content
    plan_irrelevant = state["plan"].irrelevant
    plan_chitchat = state["plan"].chitchat


    if "testcase" in plan:
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
    
    elif "testscript" in plan:

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
    
    elif "userstory" in plan:
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
    

    elif plan_chitchat:

        state["messages"].append(AIMessage(content=plan_chitchat))
        return state

    else:
        state["messages"].append(AIMessage(content=plan_irrelevant))
        return state
