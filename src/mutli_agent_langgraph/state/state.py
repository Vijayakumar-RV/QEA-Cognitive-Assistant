from typing_extensions import TypedDict, List, Literal, Optional
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import AnyMessage,BaseMessage
from src.mutli_agent_langgraph.state.planner import PlannerOutput

class State(TypedDict):

    "Represents the structure of current state used in graph"

    messages:Annotated[List[AnyMessage],add_messages]
    session_id : str
    plan:PlannerOutput
    retrived_content:Optional[str]
    testcase:Optional[str]
    testscript:Optional[str]
    userstory:Optional[str]
    document_text: Optional[str]
    embedding_enabled: Optional[bool]
    user_query: Optional[str]
