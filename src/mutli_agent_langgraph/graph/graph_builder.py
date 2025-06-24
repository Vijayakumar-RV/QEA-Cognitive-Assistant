from langgraph.graph import StateGraph , START, END
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.nodes.qea_assistant_node import QEAAssistantChatbot


class GraphBuilder:

    def __init__(self,model):
        self.llm = model
        self.graph_builder = StateGraph(State)
    
    def qea_assistant_chatbot(self):

        """
        QEA Assistant

        This intelligent assistant is designed to support Quality Engineering and Assurance (QEA) workflows by automating the creation of both manual test cases and automated test scripts.

        Capabilities:
        -------------
        1. Manual Test Case Generation:
        - Generates well-structured manual test cases for a wide range of scenarios across domains.
        - Supports both standard format and Gherkin syntax (Given-When-Then).
        - Understands functional, UI, and end-to-end scenarios based on user-provided requirements or user stories.

        2. Automated Test Script Generation:
        - Generates executable test scripts tailored to various automation tools and languages, including:
            - Selenium (Java, Python, .NET)
            - Cypress (JavaScript, TypeScript)
        - Handles multi-step flows, locator integration, and domain-specific business rules.

        3. Domain-Agnostic and Extensible:
        - Designed to adapt across industries (e-commerce, banking, healthcare, etc.).
        - Supports cross-platform test generation (web, desktop, mobile with future scalability).

        Use Cases:
        ----------
        - Create test cases from requirement descriptions or acceptance criteria.
        - Convert manual test cases to automation scripts.
        - Execute test scripts (in future integrations with automation frameworks).

        Note:
        -----
        This assistant leverages LLMs and domain knowledge to generate test artifacts contextually. It can be integrated into CI/CD pipelines, QA platforms, or used interactively through a conversational interface.
        """

        self.qea_chatbot = QEAAssistantChatbot(self.llm)

        self.graph_builder.add_node("chatbot",self.qea_chatbot.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)

    def setup_graph(self,usecase:str):

        """
        Sets up the graph for the selected usecase
        """
        print(usecase)
        if usecase == "QEA_Assistant":
            self.qea_assistant_chatbot()

        return self.graph_builder.compile()


