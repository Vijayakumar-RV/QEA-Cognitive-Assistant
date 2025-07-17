import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,BaseMessage
import json
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation
import time
class DisplayResultStreamlit:
    def __init__(self,usecase,graph,user_message,session_id):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.session_id = session_id


    def _render_message(self, msg: BaseMessage):
        """Render any Chat / Tool message in Streamlit chat UI."""
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)

        # show assistant replies **only** if they have no tool_calls
        elif msg.type == "ai" and not getattr(msg, "tool_calls", None):
            with st.chat_message("assistant"):
                st.markdown(msg.content)


    def disply_result_on_ui(self):

        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message
        session_id = self.session_id

        if usecase == "QEA_Assistant":
            conversation_memory = LangchainConversation(session_id=session_id)
            for old_msg in conversation_memory.get_conversation_memory().load_memory_variables({})["history"]:
                self._render_message(old_msg)
            
            if user_message:
                with st.chat_message("user"):
                    st.markdown(user_message)
                
                assistant_block = st.chat_message("assistant")
                stream_area = assistant_block.empty()

                state_input ={
                    "messages":[HumanMessage(content=user_message)],
                    "session_id": session_id
                }
           
            with st.spinner("🤖 Generating response..."):
                for event in graph.stream(state_input):
                    print(event.values())
                    for value in event.values():
                        if "messages" not in value:
                            continue
                        last_msg = value["messages"][-1]

                        # Stream assistant text incrementally
                        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                            continue

                        if isinstance(last_msg, AIMessage):
                            stream_area.markdown(last_msg.content)