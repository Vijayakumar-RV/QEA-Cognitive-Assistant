from src.mutli_agent_langgraph.state.state import State


class QEAAssistantChatbot:

    def __init__(self,model):
        self.llm = model

    def process(self,state: State)-> dict :
        """
        Process the input state and generate a qea chatbot response
        """
        print(state["messages"])
        print(f"llm selected: {self.llm}")
        return {"messages":self.llm.invoke(state["messages"])}