from langgraph.graph import StateGraph , START, END
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.nodes.qea_assistant_node import QEAAssistantChatbot
from src.mutli_agent_langgraph.nodes.qea_research_node import QEARESEARCHNODE
from src.mutli_agent_langgraph.agents.planner_agent import Planner
from src.mutli_agent_langgraph.agents import retriever_agent
from src.mutli_agent_langgraph.tools.research_tools import get_tools,create_tool_node
from src.mutli_agent_langgraph.nodes.qea_document_node import DocumentAnalyzerNode
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import run_mlflow_run
from langgraph.prebuilt import ToolNode, tools_condition
import traceback

from src.mutli_agent_langgraph.guardrails.node import create_guardrails_node, route_after_guardrails
from src.mutli_agent_langgraph.routers.planner_router import route_after_planner

class GraphBuilder:

    def __init__(self, model, temperature,enable_judge:bool= False,test_case_format:str= "",test_script_lang:str= "",test_framework:str= ""):
        self.llm = model
        self.temperature = temperature
        self.enable_judge = enable_judge
        self.test_case_format = test_case_format
        self.test_script_lang = test_script_lang
        self.test_framework = test_framework
        self.graph_builder = StateGraph(State)


    def assistant_chatbot(self):
        """
        START → guardrails ─┬→ (to_end) END
                            └→ (to_planner) planner ─┬→ (to_retriver) retriver → qeacognitive → END
                                                      └→ (to_qeacognitive) qeacognitive → END
        """
        planner = Planner(self.llm)
        qea_chatbot = QEAAssistantChatbot(self.llm, self.temperature,self.test_case_format,self.test_script_lang,self.test_framework)

        self.graph_builder.add_node("guardrails", create_guardrails_node())
        self.graph_builder.add_node("planner", planner.planner_agent)
        self.graph_builder.add_node("retriver", retriever_agent.retriever_agent)
        self.graph_builder.add_node("qeacognitive", qea_chatbot.process)

        # 1) Guardrails first
        self.graph_builder.add_edge(START, "guardrails")
        self.graph_builder.add_conditional_edges(
            "guardrails",
            route_after_guardrails,
            {
                "to_planner": "planner",
                "to_end": END,
            }
        )

        # 2) After planner, decide if we need retrieval or can go straight to QEA
        self.graph_builder.add_conditional_edges(
            "planner",
            route_after_planner,
            {
                "to_retriver": "retriver",
                "to_qeacognitive": "qeacognitive",
            }
        )

        # 3) Continue
        self.graph_builder.add_edge("retriver", "qeacognitive")

        #Judge eval
        if self.enable_judge:
            from src.mutli_agent_langgraph.nodes.judge_node import create_judge_node
            self.graph_builder.add_node("judge", create_judge_node())
            self.graph_builder.add_edge("qeacognitive", "judge")
            self.graph_builder.add_edge("judge", END)
        else:
            self.graph_builder.add_edge("qeacognitive", END)

    def research_assistant(self):
        tools = get_tools()
        tool_node = create_tool_node(tools)
        llm = self.llm
        chatbot_object = QEARESEARCHNODE(llm, self.temperature)
        qea_research_chatbot = chatbot_object.process(tools=tools)

        self.graph_builder.add_node("chatbot", qea_research_chatbot)
        self.graph_builder.add_node("tools", tool_node)

        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools", "chatbot")

    def document_analyzer(self):
        analyzer = DocumentAnalyzerNode(self.llm, self.temperature)
        self.graph_builder.add_node("analyzer", analyzer.process)
        self.graph_builder.add_edge(START, "analyzer")
        self.graph_builder.add_edge("analyzer", END)

    def setup_graph(self, usecase: str):
        print(usecase)
        try:
            if usecase == "QEA_Assistant":
                self.assistant_chatbot()
            if usecase == "QEA Research Assistant":
                self.research_assistant()
            if usecase == "QEA Document Assistant":
                self.document_analyzer()
            return self.graph_builder.compile()
        except Exception:
            traceback.print_exc()
            raise
