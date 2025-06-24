import streamlit as st
import os
from src.mutli_agent_langgraph.ui.streamlit.ui_configfile import Config
import uuid
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation

class LoadStreamlitUI:

    def __init__(self):
        self.config = Config()
        self.user_controls ={}
        self.session_id = None
    

    def load_streamlit_ui(self):
        st.set_page_config(page_title=self.config.get_title(), layout="wide", initial_sidebar_state="expanded")
        st.header("🧠" + self.config.get_title())

        if "session_id" not in st.session_state:
            st.session_state["session_id"] = str(uuid.uuid4())

        self.session_id = st.session_state["session_id"]
        

        with st.sidebar:
            #get option from the config file
            llm_option = self.config.get_llm_options()
            usecase_option = self.config.get_usecase_options()

            #LLM Selection
            self.user_controls["select_llm"] = st.selectbox("Select LLM", llm_option, index=0)

            if self.user_controls["select_llm"] == "OpenAI":
                model_options = self.config.get_openai_model()
                self.user_controls["Selected_OpenAI_Model"] = st.selectbox("Select OpenAI Model", model_options, index=0)

            elif self.user_controls["select_llm"] == "GROQ_AI":
                model_options = self.config.get_groq_model()
                self.user_controls["Selected_Groq_Model"] = st.selectbox("Select Groq Model", model_options, index=0)
                self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = st.text_input("Groq API Key", type="password")

            elif self.user_controls["select_llm"] == "OLLAMA":
                model_options = self.config.get_ollama_model()
                self.user_controls["Selected_Ollama_Model"] = st.selectbox("Select Ollama Model", model_options, index=0)
            
            if not self.user_controls["select_llm"]:
                st.warning("⚠️ Please select an LLM option.")
            
            #Usecase Selection

            self.user_controls["select_usecase"] = st.selectbox("Select Usecase", usecase_option, index=0)

            if self.user_controls["select_llm"]=="OpenAI" and self.user_controls["Selected_OpenAI_Model"]=="o4-mini":

                st.slider("Temperature",disabled=True)
            else:
                self.user_controls["select_temperature"] = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)


        

        self.user_controls["session_id"] = self.session_id

        return self.user_controls
    