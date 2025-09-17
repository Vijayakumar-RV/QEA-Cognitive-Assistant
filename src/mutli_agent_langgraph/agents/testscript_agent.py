from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from src.mutli_agent_langgraph.state.state import State
import re
import time
import hashlib
from src.mutli_agent_langgraph.utils.session_store import load_session_rows
_TC_ID_RE = re.compile(r"\b(Test Case ID|TC[-\s]*ID)\b[^A-Za-z0-9]*([A-Za-z0-9_-]+)", re.IGNORECASE)

def _infer_script_id(test_case_markdown: str) -> str:
    if not test_case_markdown:
        return f"script_{int(time.time())}"
    m = _TC_ID_RE.search(test_case_markdown)
    if m and m.group(2):
        return m.group(2)
    # fallback: stable hash of first 200 chars
    h = hashlib.sha1((test_case_markdown[:200] or "").encode("utf-8")).hexdigest()[:10]
    return f"script_{h}"


def testscript(state:State,session_id,retriver,user_message,model,test_script_lang:str,test_framework:str)->State:
    test_case = load_session_rows(session_id)

    if  test_case:

        test_script_prompt = f"""
        You are a Test Script Generator Agent.

        Task:
        Write a ** {test_script_lang} ** ** {test_framework} ** test script implementing ONLY the steps and locators provided in the test case and context.

        Guardrails & Constraints:
        - DO NOT invent new steps, element locators, actions, or assertions.
        - Use step order, field names, and locators exactly as provided.
        - If an element locator or step is missing, output: "# ERROR: Step or locator missing: [describe missing piece]" at the appropriate place in the code.
        - Properly add waits, error handling, or logic that are industry standard.
        - Do NOT make assumptions about the application flow.
        - If insufficient information is provided to produce a script, output: "# ERROR: Insufficient context to generate test script."
        - Use the Test case provided and Retrived Context and create the test script

        Output Format:
        Output only valid code for {test_script_lang} + {test_framework} (no explanations).

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
        Write a ** {test_script_lang} ** ** {test_framework} ** test script implementing ONLY the steps and locators provided in the test case and context.

        Guardrails & Constraints:
        - DO NOT invent new steps, element locators, actions, or assertions.
        - Use step order, field names, and locators exactly as provided.
        - If an element locator or step is missing, output: Suggest an best alternate Locator.
        - Properly add waits, error handling, or logic that are industry standard.
        - Do NOT make assumptions about the application flow.
        - If insufficient information is provided to produce a script, output: "# ERROR: Insufficient context to generate test script."
        - Use the Test case provided and Retrived Context and create the test script
        - Include maximizing the window size before performing actions.
        - Generate test script with preconditions.

        Output Format:
        Output only valid code for {test_script_lang} + {test_framework} (no explanations).

        Test Case:
        \"\"\"{test_case}\"\"\"

        Element Locators/Context:
        \"\"\"{retriver}\"\"\"

        User Query:

        \"\"\"{user_message}"\"\"
        """
    
    context = [
        SystemMessage(content=test_script_prompt),
        HumanMessage(content=user_message)
    ]

    response = model.invoke(context)
    # response = response.content
    #     # Remove any Markdown fences
    # if response.startswith("```"):
    #     response = response.split("```", 2)[1]  # take middle part
    # # Remove common leading "python" / "javascript" hints
    # response = response.replace("python\n", "").replace("javascript\n", "")

    # # Remove docstring-like triple quotes
    # import re
    # response = re.sub(r'"""[\s\S]*?"""', "", response)
    # response = re.sub(r"'''[\s\S]*?'''", "", response)

    # # Replace <MISSING: …> with TODO comments
    # response = re.sub(r"<MISSING:[^>]+>", "# TODO: add missing locator", response)

    state["testscript"] = response.content
    state["messages"].append(AIMessage(content=response.content))

    
    return state