from langgraph.graph import StateGraph , START, END
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.nodes.qea_assistant_node import QEAAssistantChatbot
from src.mutli_agent_langgraph.nodes.qea_research_node import QEARESEARCHNODE
from src.mutli_agent_langgraph.agents.planner_agent import Planner
from src.mutli_agent_langgraph.agents import retriever_agent
from src.mutli_agent_langgraph.tools.research_tools import get_tools,create_tool_node
from src.mutli_agent_langgraph.nodes.qea_document_node import DocumentAnalyzerNode

from langgraph.prebuilt import ToolNode,tools_condition
import traceback
class GraphBuilder:

    def __init__(self,model):
        self.llm = model
        self.graph_builder = StateGraph(State)


    def assistant_chatbot(self):

        """
        QEA Assistant

        This intelligent assistant is designed to support Quality Engineering and Assurance (QEA) workflows by automating the creation of both manual test cases and automated test scripts.
        It constructs a chatbot graph consisting of the chatbot node and a tool node. The tool node is responsible for retrieving contextual or domain-specific knowledge from a vector or document database, enhancing the assistant’s response accuracy.

        This method creates a chatbot graph that includes both the chatbot node and a tool node. It defines tools, initializes the chatbot with tool capabilities, and sets up conditional and direct
        edges between nodes. The chatbot node is set as the entry point.

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
        - Retrieve knowledge from previous documentation or historical test data.
        - Execute test scripts (in future integrations with automation frameworks).

        Note:
        -----
        This assistant leverages LLMs, embedded domain knowledge, and retrieval-augmented generation (RAG) through the tool node to generate test artifacts contextually. It can be integrated into CI/CD pipelines, QA platforms, or used interactively through a conversational interface.
        """

        
        planner = Planner(self.llm)
        qea_chatbot = QEAAssistantChatbot(self.llm)
        

        self.graph_builder.add_node("planner",planner.planner_agent)

        self.graph_builder.add_node("retriver",retriever_agent.retriever_agent)
        
        self.graph_builder.add_node("qeacognitive",qea_chatbot.process)
        
        self.graph_builder.add_edge(START,"planner")

        self.graph_builder.add_edge("planner","retriver")

        self.graph_builder.add_edge("retriver","qeacognitive")

        self.graph_builder.add_edge("qeacognitive",END)

    def research_assistant(self):
        """
        Builds an advanced agentic graph with tool integration.
        This method creates a graph that includes both a chatbot node and a tool node.
        It defines the tools, initialize the chatbot with tool capabilities, and sets up
        conditional and direct edges between nodes. The chatbot is set as an entry point.
        """

        tools = get_tools()
        tool_node = create_tool_node(tools)

        llm = self.llm

        chatbot_oject = QEARESEARCHNODE(llm)
        qea_research_chatbot = chatbot_oject.process(tools=tools)

        self.graph_builder.add_node("chatbot",qea_research_chatbot)
        self.graph_builder.add_node("tools",tool_node)
        
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_conditional_edges("chatbot",tools_condition)
        self.graph_builder.add_edge("tools","chatbot")
        
    def document_analyzer(self):

        """
        Sets up the QEA Document Analyzer Node.

        This node handles document summarization and question answering.
        It supports multiple file formats like PDF, DOCX, TXT, CSV, PPTX, and images (OCR).
        
        Based on user input:
        - If a question is asked, it answers using the document (with or without embedding).
        - If no question is asked, it returns a summary of the document.
        """

        analyzer = DocumentAnalyzerNode(self.llm)
        self.graph_builder.add_node("analyzer", analyzer.process)
        self.graph_builder.add_edge(START, "analyzer")
        self.graph_builder.add_edge("analyzer", END)

    
    def setup_graph(self,usecase:str):

        """
        Sets up the graph for the selected usecase
        """
        print(usecase)
        try:
            if usecase == "QEA_Assistant":
                self.assistant_chatbot()
            if usecase == "QEA Research Assistant":
                self.research_assistant()
            if usecase == "QEA Document Assistant":
                self.document_analyzer()

            return self.graph_builder.compile()
        except Exception as e:
            traceback.print_exc()
            raise
        


