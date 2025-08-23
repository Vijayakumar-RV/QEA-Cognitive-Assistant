from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.utils.test_clean import parse_json_loose, normalize_test_cases, validate_cases, cases_to_memory, cases_to_markdown,cases_to_rows
import re
from src.mutli_agent_langgraph.agents.save_execute_agent import save_testcase_excel_csv,_first_heading
def testcase(retriver,user_message,model,state:State)->State:

    print(f"retriver in testcase {retriver}")
    test_case_prompt = f"""
            You are a Test Case Generator Agent for software quality engineering.

            Task:
            Generate a clear, detailed, and step-by-step manual test case for the scenario request by the user, using ONLY the provided context (UI flow, user story, or requirements).

            Guardrails & Constraints:
            - Use only the UI elements and labels exactly as described in the context.
            - Provide detailed test steps for the **test cases**. Follow test design techniques while creating test cases.
            - If an expected step, element, or navigation is clearly described, include it in the test case.
            - If a step or element is not explicitly mentioned, do not add it to the test case.
            - Do not invent UI components or actions that are not present in the given context.
            - If there is truly no way to proceed, provide the best possible partial steps and flag missing information.
            - Ensure ALL test cases are output as a **single valid JSON array**. Do not output multiple top-level JSON objects.
            - Do not include any explanations, comments, markdown code fences, or text outside the JSON array.
            
            Output Format:
            You must output the test case in JSON format with the following fields:
            - Test Case ID
            - Test Case Title: <Title>
            - Description
            - Preconditions
            - Test Steps (as a list of steps)
            - Expected Result
            - Test Type (Manual or Automation)
            - Priority (Low, Medium, High, Critical)
            - Tags (optional)

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

    print(f"response for open ai {response.content}")

    parsed = parse_json_loose(response.content)
    if not parsed:
        print("No valid JSON found.")
        return state
    
    cases = normalize_test_cases(parsed)

    errs = validate_cases(cases)
    if errs:
        # choose: log somewhere, or append a warning message
        pass

    rows = cases_to_rows(cases)
    mark_down =  cases_to_markdown(cases)
    mem = cases_to_memory(cases,5,3)
    state["testcase"] = mark_down
    state["testcase_mem"] = mem
    state["test_rows"] = rows
    state["messages"].append(AIMessage(content=mark_down))
   
    return state