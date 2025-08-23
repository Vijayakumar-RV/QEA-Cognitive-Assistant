import streamlit as st
import os
from src.mutli_agent_langgraph.ui.streamlit.ui_configfile import Config
import uuid
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation
from dotenv import load_dotenv
from src.mutli_agent_langgraph.tools.document_analyzer_tools import parse_document
import time
import json
import pandas as pd

class LoadStreamlitUI:

    def __init__(self):
        self.config = Config()
        self.user_controls ={}
        self.session_id = None
        
    

    def load_streamlit_ui(self):
        load_dotenv()
        st.set_page_config(page_title=self.config.get_title(), layout="wide", initial_sidebar_state="expanded")
        st.header("🧠" + self.config.get_title())

        if "session_id" not in st.session_state:
            st.session_state["session_id"] = str(uuid.uuid4())

        self.session_id = st.session_state["session_id"]
        

        with st.sidebar:
            #get option from the config file
            llm_option = self.config.get_llm_options()
            usecase_option = self.config.get_usecase_options()
            api_key_option = self.config.get_api_key()

            #LLM Selection
            self.user_controls["select_llm"] = st.selectbox("Select LLM", llm_option, index=0)

            if self.user_controls["select_llm"] == "OpenAI":
                model_options = self.config.get_openai_model()
                self.user_controls["Selected_OpenAI_Model"] = st.selectbox("Select OpenAI Model", model_options, index=0)
                self.user_controls["Select_API_Key"] = st.selectbox("Select API Key",api_key_option,index=0)
                if self.user_controls["Select_API_Key"] != "Default":
                    self.user_controls["OPEN_AI_KEY"] = st.session_state["OPENAI_API_KEY"] = st.text_input("Open API Key", type="password")
                else:
                    self.user_controls["OPEN_AI_KEY"] = st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


            elif self.user_controls["select_llm"] == "GROQ_AI":
                model_options = self.config.get_groq_model()
                self.user_controls["Selected_Groq_Model"] = st.selectbox("Select Groq Model", model_options, index=0)
                self.user_controls["Select_API_Key"] = st.selectbox("Select API Key",api_key_option,index=0)
                if self.user_controls["Select_API_Key"] != "Default":
                    self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = st.text_input("Groq API Key", type="password")
                else:
                    self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

            elif self.user_controls["select_llm"] == "GOOGLE_AI":
                model_options = self.config.get_google_model()
                self.user_controls["Selected_Google_Model"] = st.selectbox("Select Google Model", model_options, index=0)
                self.user_controls["Select_API_Key"] = st.selectbox("Select API Key",api_key_option,index=0)
                if self.user_controls["Select_API_Key"] != "Default":
                    self.user_controls["GOOGLE_API_KEY"] = st.session_state["GOOGLE_API_KEY"] = st.text_input("Google API Key", type="password")
                    os.environ["GOOGLE_API_KEY"] = self.user_controls["GOOGLE_API_KEY"]
                    
                else:
                   self.user_controls["GOOGLE_API_KEY"] = st.session_state["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
                   os.environ["GOOGLE_API_KEY"] = self.user_controls["GOOGLE_API_KEY"]
                   print(f"user key from UI : {self.user_controls["GOOGLE_API_KEY"]}")

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
            
            self.user_controls["enable_judge"] = st.toggle("Enable Judge (Eval mode)", value=False)

            tc_style = st.selectbox(
                "Test Case Format",
                options=["default", "gherkin"],
                index=0,
                help="Choose 'gherkin' to get Given/When/Then scenarios."
            )

            script_lang = st.selectbox(
                "Script Language",
                options=["python", "javascript", "java"],
                index=0
            )

            framework = st.selectbox(
                "Automation Framework",
                options=["selenium", "cypress", "playwright"],
                index=0
            )

            # Simple compatibility guardrails (keeps UX friendly)
            compat = {
                "selenium": {"python", "javascript", "java"},
                "cypress": {"javascript"},
                "playwright": {"python", "javascript", "java"},
            }
            
            if script_lang not in compat[framework]:
                st.warning(f"{framework.title()} works with {sorted(list(compat[framework]))}. "
                        f"Auto-switching language to a compatible option.")
                # Pick first compatible lang deterministically
                script_lang = sorted(list(compat[framework]))[0]

            # Persist into user_controls the SAME way you store existing knobs
            self.user_controls["tc_style"] = tc_style
            self.user_controls["script_lang"] = script_lang
            self.user_controls["framework"] = framework

            if self.user_controls["select_usecase"] =="QEA Research Assistant":
                self.user_controls["TAVILY_API_KEY"] = st.selectbox("TAVILY API KEY",api_key_option,index=0)
                if self.user_controls["TAVILY_API_KEY"] != "Default":
                    os.environ["TAVILY_API_KEY"] = self.user_controls["TAVILY_API_KEY"] = st.session_state["TAVILY_API_KEY"] = st.text_input("TAVILY API KEY",type="password")
                else:
                    os.environ["TAVILY_API_KEY"] = self.user_controls["TAVILY_API_KEY"]= st.session_state["TAVILY_API_KEY"]= os.getenv("TAVILY_API_KEY")

            if self.user_controls["select_usecase"] =="QEA Document Assistant":
                
                self.user_controls["upload_file"]=uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "pptx", "txt", "csv", "jpeg", "jpg", "png", "json"])
                self.user_controls["enable_embedding"]=enable_embedding = st.toggle("Enable Smart Retrieval (Embedding)")

                if "document_text" not in st.session_state:
                    st.session_state["document_text"] = None
                if "embedded_store" not in st.session_state:
                    st.session_state["embedded_store"] = None
                if "last_summary" not in st.session_state:
                    st.session_state["last_summary"] = ""
                if "last_qa" not in st.session_state:
                    st.session_state["last_qa"] = ""

                LOG_PATH = "upload_log.txt"

                def log_upload(file_name, file_type, size):
                    with open(LOG_PATH, "a") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Uploaded: {file_name}, Type: {file_type}, Size: {size} bytes\n")

                def display_upload_history():
                    if os.path.exists(LOG_PATH):
                        with open(LOG_PATH, "r") as f:
                            logs = f.readlines()
                        st.expander("📜 Upload History").write("".join(logs))

                if uploaded_file:
                    content, file_type = parse_document(uploaded_file)
                    st.session_state["document_text"] = content

                    st.info(f"**File name:** {uploaded_file.name}\n\n**Type:** {uploaded_file.type}\n\n**Size:** {uploaded_file.size / 1024:.2f} KB")
                    # log_upload(uploaded_file.name, uploaded_file.type, uploaded_file.size)
                    # display_upload_history()

                    st.subheader("📄 Document Preview")
                    if uploaded_file.name.endswith(".csv"):
                        try:
                            df = pd.read_csv(uploaded_file)
                            st.dataframe(df)
                        except:
                            st.text_area("Text Preview", content[:3000])
                    elif uploaded_file.name.endswith(".json"):
                        try:
                            data = json.loads(content)
                            st.json(data)
                        except:
                            st.text_area("Text Preview", content[:3000])
                    else:
                        st.text_area("Text Preview", content[:3000])

                    if st.button("Analyze with AI"):
                        with st.spinner("Processing..."):
                            self.user_controls["document_text"] = content
                            self.user_controls["embedding_enabled"] = enable_embedding
                            self.user_controls["analyze_triggered"] = True

        self.user_controls["session_id"] = self.session_id

        return self.user_controls
    