import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import json
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation

class DisplayResultStreamlit:
    def __init__(self,usecase,graph,user_message,session_id):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.session_id = session_id

    def disply_result_on_ui(self):

        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message
        session_id = self.session_id

        if usecase == "QEA_Assistant":
            conversation_memory = LangchainConversation(session_id=session_id)
            memory = conversation_memory.get_conversation_memory()
            history = memory.load_memory_variables({})["history"]

            #Display the conversation history
            if history:
                st.subheader("Conversation History")
                for message in history:
                    if message.type == "human":
                        with st.chat_message("user"):
                            st.markdown(f"**User:** {message.content}")
                    if message.type == "assistant":
                        with st.chat_message("assistant"):
                            st.markdown(f"**Assistant:** {message.content}")
            
            if user_message:
                with st.chat_message("user"):
                    st.markdown(user_message)
                
                assistant_block = st.chat_message("assistant")
                stream_area = assistant_block.empty()

                state_input ={
                    "messages":[{"type":"human","content":user_message}],
                    "session_id": session_id
                }
           

            for event in graph.stream(state_input):
                print(event.values())
                for value in event.values():
                    if "messages" in value:
                        last_message = value["messages"][-1]
                        if last_message["type"] == "assistant":
                            stream_area.markdown(f"**Assistant:** {last_message['content']}")
                            