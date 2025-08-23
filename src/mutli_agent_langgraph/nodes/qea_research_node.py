from src.mutli_agent_langgraph.state.state import State

class QEARESEARCHNODE:

    def __init__(self,model,temperature):
        self.llm=model
        self.temperature=temperature

    def process(self,tools):
        """
        Returns the chatbot node function
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state:State):

            """
            chatbot logic for processing the input state and returning the a response
            """
            
            return {"messages":[llm_with_tools.invoke(state["messages"])]}
        
        return chatbot_node


