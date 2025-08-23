from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.utils.test_clean import parse_json_loose, normalize_test_cases, validate_cases, cases_to_memory, cases_to_markdown,cases_to_rows
from src.mutli_agent_langgraph.agents.save_execute_agent import save_testcase_excel_csv,_first_heading
from src.mutli_agent_langgraph.utils.test_clean import repair_gherkin_in_json,normalize_gherkin_multiline


def testcase(retriver,user_message,model,state:State,test_case_format:str)->State:

    print(f"Test Case Format: {test_case_format}")
    # print(f"retriver in testcase {retriver}")
    
    test_case_prompt = f"""
            You are a Test Case Generator Agent for software quality engineering.

            You will generate manual test cases using ONLY the "Provided Context" and the "User Query".
            You must follow ALL rules below without exception.

            =====================
            MODE
            =====================
            - Mode is: {test_case_format}     # "default" or "gherkin"
            - Behavior:
            - If "default": produce standard step-by-step test cases.
            - If "gherkin": produce BDD Gherkin content, but STILL return everything inside a single JSON array per the schema below; the Gherkin text MUST be a single multi-line string under "Test Steps".

            =====================
            GUARDRAILS & CONSTRAINTS
            =====================
            - Use only UI elements and labels exactly as they appear in the context.
            - Apply test design techniques (equivalence classes, boundaries, negatives) within the given context.
            - Do not invent steps or elements not present in the context.
            - If info is missing, provide the best partial case and mark placeholders like "<MISSING: …>".
            - Output MUST be a single valid JSON array (no text outside, no markdown, no comments).

            =====================
            OUTPUT SCHEMA (ALWAYS JSON)
            =====================
            Return a single JSON array of one or more test case objects. Each object MUST contain EXACTLY these fields:

            1) "Test Case ID"        : string
            2) "Test Case Title"     : string
            3) "Description"         : string
            4) "Preconditions"       : string or array of strings
            5) "Test Steps"          :
            - If mode == "default": array of strings (each item is one executable step).
            - If mode == "gherkin": a SINGLE multi-line string containing a valid Gherkin feature.
                • Must start with: "Feature: <short feature name>"
                • Include at least one "Scenario: <name>"
                • **FORMATTING CONTRACT (STRICT):**
                    - Every Gherkin keyword MUST start on a NEW line at the start of the line.
                    - Allowed keywords: "Background:", "Scenario:", "Scenario Outline:", "Given", "When", "Then", "And", "But", "|".
                    - **Never combine multiple keywords on the same line.**
                    - Do not join clauses with commas or "and"; create separate lines.
                    - Preserve newline characters `\n` between lines. No wrapping.
                • **Tiny example (format only, not content):**
                    Feature: Login
                    Scenario: Valid login
                    Given the user is on the Login page
                    When the user enters a valid email
                    And the user enters a valid password
                    And the user clicks the "Login" button
                    Then the user is redirected to the My Account dashboard
            6) "Expected Result"     : string
            7) "Test Type"           : one of ["Manual","Automation"]
            8) "Priority"            : one of ["Low","Medium","High","Critical"]
            9) "Tags"                : array of strings (optional; omit if none)

            =====================
            QUALITY RULES
            =====================
            - Steps must be deterministic and executable.
            - Use exact control labels from the context (no synonyms).
            - No duplicate test cases in the same output.

            =====================
            INPUTS
            =====================
            User Query:
            \"\"\"{user_message}\"\"\"

            Provided Context:
            \"\"\"{retriver}\\n\"\"\"   # note: allow real newlines inside the context

            =====================
            RESPONSE
            =====================
            Return ONLY the JSON array. Do not include any other text.


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
    gherkin_passed = repair_gherkin_in_json(parsed, test_case_format)
    
    cases = normalize_test_cases(gherkin_passed,test_case_format)
    
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