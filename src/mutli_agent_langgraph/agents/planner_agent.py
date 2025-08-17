from langchain_core.messages import HumanMessage,SystemMessage
from langchain.output_parsers import PydanticOutputParser
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.state.planner import PlannerOutput 

class Planner:

    def __init__(self,model):
        
        self.llm = model

    def planner_agent(self,state:State)->dict:
        try:
            print(f"state['test_rows']: {state['test_rows']}")
        except KeyError:
            print("Key 'test_rows' not found in state.")
        user_message = str(state["messages"][-1].content)
        print(f"Last user message : {user_message}")
        parser = PydanticOutputParser(pydantic_object=PlannerOutput)
        format_instructions = parser.get_format_instructions()
        
        system_prompt = f""" You are a planner agent of the Tester Copilot.
            Given the follwing User request, analyze the intent of the user and output a Json action plan describing exactly what information needs to be retrieved or which agent should be called next.
            - If the user wants to see, view, display, or summarize a user story, add "userstory" to "generation".
            - Identify if you need to retrieve the UI Flow or the User Story or both. Testcase and Testscript generation requires both User story and UI flow retrival.
            - The User story consists of User Story and acceptance criteria which helps in creating test cases, test scripts and the displaying the user stories.
            - The UI Flow consists of all the pages elements,attributes,before and after page details. This helps in creating test cases and test scripts.
            - "testcase" and "testscript" are independent: include each in "generation" ONLY if the user request explicitly mentions it.
            - If generation is not needed, leave it as an empty list.
            - If retrieval is not needed, leave it as an empty list.
            - Be specific. Do not hallucinate or invent new steps or elements or locators.
            - If the message is unrelated to the application (such as general knowledge, capitals, weather, etc.), populate the ouput in irrelevant, mentioin that you are QEA bot.
            - For chit-chat or unrelated questions, reply with just the appropriate friendly or fallback message, poulate the ouput in chitchat.
            - If the user wants to save test cases or test scripts, populate the output in save_testcases or save_execute_testscripts.
            **IMPORTANT:**  
            
            - Output ONLY in the following format.
            {format_instructions}


            Example 1:
            User: "Show me the user story for login"
            Output:
                {{
                    "reterivals": [{{"type": "User_Story"}}],
                    "generation": ["userstory"],
                    "irrelevant": null,
                    "chitchat": null,
                    "user_request": "Show me the user story for login"
                }}

            Example 2:
            User: "Generate a detailed test case for the user story US-101"
            Output:
            {{
                "reterivals": [{{"type": "both"}}],
                "generation": ["testcase"],
                "irrelevant": null,
                "chitchat": null,
                "user_request": "Generate a detailed test case for the user story US-101"
            }}

            Example 3:
            User: "Retrieve user story US-102 and generate a test script for it"
            Output:
            {{
                "reterivals": [{{"type": "both"}}],
                "generation": ["testscript"],
                "irrelevant": null,
                "chitchat": null,
                "user_request": "Retrieve user story US-102 and generate a test script for it"
            }}

            Example 4:
            User: "I want both a test case and a test script for the login functionality"
            Output:
            {{
                "reterivals": [{{"type": "both"}}],
                "generation": ["testcase", "testscript"],
                "irrelevant": null,
                "chitchat": null,
                "user_request": "I want both a test case and a test script for the login functionality"
            }}

            Example 5:
            User: "Hi there!"
            Output:
            {{
                "reterivals": [],
                "generation": [],
                "irrelevant": null,
                "chitchat": "Hello! How can I help you with application functionalities or test generation today?",
                "user_request": "Hi there!"
            }}

            Example 6:
            User: "What is the capital of France?"
            Output:
            {{
                "reterivals": [],
                "generation": [],
                "irrelevant": "I don't have enough information on that. Please ask me something related to the application and its functionalities.",
                "chitchat": null,
                "user_request": "What is the capital of France?"
            }}

            Example 7:
            User: "Please save the test cases in CSV format."
            Output:
            {{
                "reterivals": [],
                "generation": [],
                "irrelevant": null,
                "chitchat": null,
                "save_testcases": ["csv"],
                "save_execute_testscripts": null,
                "user_request": "Save the test cases in CSV format."
            }}

            Example 8:
            User: "I want to execute the test scripts."
            Output:
            {{
                "reterivals": [],
                "generation": [],
                "irrelevant": null,
                "chitchat": null,
                "save_testcases": null,
                "save_execute_testscripts": ["execute"],
                "user_request": "I want to execute the test scripts."
            }}

        """
        
        context = [

            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        planner_output = self.llm.invoke(context)
        try:
            plan_parser = parser.parse(planner_output.content)
            state["plan"] = plan_parser

            print(f"Final plan {state["plan"]}")
        except Exception as e:
            print(f"Error in planner agent: {e}")
            state["plan"] = None
        return state
