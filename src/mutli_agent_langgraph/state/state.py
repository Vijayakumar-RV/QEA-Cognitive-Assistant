from typing_extensions import TypedDict, List, Literal, Optional
from langgraph.graph.message import add_messages
from typing import Annotated, Dict, Any
from langchain_core.messages import AnyMessage,BaseMessage
from src.mutli_agent_langgraph.state.planner import PlannerOutput

class State(TypedDict,total=False):

    "Represents the structure of current state used in graph"

    messages:Annotated[List[BaseMessage],add_messages]
    session_id : str
    plan:PlannerOutput
    retrived_content:Optional[str]
    testcase:Optional[str]
    testcase_mem: Optional[str]
    test_rows: Optional[List[Dict[str, Any]]] 
    testscript:Optional[str]
    userstory:Optional[str]
    document_text: Optional[str]
    embedding_enabled: Optional[bool]
    user_query: Optional[str]
    safety: Dict[str, Any]               # <- optional
    retrieved_context: str 
