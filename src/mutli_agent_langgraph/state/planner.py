from typing_extensions import Literal,List,Optional
from pydantic import BaseModel


class RetreivalItem(BaseModel):
    type: Literal["UI_Flow","User_Story","both"]

class PlannerOutput(BaseModel):
    reterivals: List[RetreivalItem]
    generation: Optional[List[Literal["userstory","testcase","testscript"]]]=None
    irrelevant: Optional[str]=None
    chitchat : Optional[str]=None
    save_testcases: Optional[List[Literal["csv","xlsx"]]]=None
    save_execute_testscripts: Optional[List[Literal["save","run","both"]]]=None
