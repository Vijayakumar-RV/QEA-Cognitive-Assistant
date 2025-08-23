import streamlit as st
from src.mutli_agent_langgraph.ui.streamlit.load_ui import LoadStreamlitUI
from src.mutli_agent_langgraph.LLMS.groqllm import GroqLLM
from src.mutli_agent_langgraph.LLMS.ollamallm import OllamaLLM
from src.mutli_agent_langgraph.LLMS.openaillm import OpenAILLM
from src.mutli_agent_langgraph.LLMS.googlellm import GoogleLLM
from src.mutli_agent_langgraph.graph.graph_builder import GraphBuilder
from src.mutli_agent_langgraph.ui.streamlit.display_results import DisplayResultStreamlit
from src.mutli_agent_langgraph.ui.streamlit.ui_configfile import Config
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import init_mlflow
from huggingface_hub import login
import os
def load_multi_agent_langgraph_ui():
    """
    Load the Streamlit UI for the Multi-Agent LangGraph application.
    This function initializes the LoadStreamlitUI class and loads the UI components
    based on the configuration settings defined in the Config class.
    Returns:
        dict: A dictionary containing user controls for the Streamlit UI.
    """
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    login(token=hf_token)
    if hf_token:
        try:    
            print("[HF login] successful")
        except Exception as e:
            print(f"[HF login] skipped: {e}")

        # --- MLflow boot (safe if called once) ---
    cfg = Config()
    try:
        if cfg.getboolean("tracking", "enable_mlflow", fallback=True):
            init_mlflow(
                tracking_uri=cfg.get("tracking", "mlflow_tracking_uri", fallback="./mlruns"),
                experiment_name=cfg.get("tracking", "experiment_name", fallback="QEA_Copilot"),
            )
    except Exception as e:
        # Non-fatal: continue without tracking if misconfigured
        print(f"[MLflow] init skipped: {e}")
    
    # Initialize the LoadStreamlitUI class
    load_ui = LoadStreamlitUI()
    

    
    user_input = load_ui.load_streamlit_ui()

    

    if not user_input:
        st.warning("⚠️ Please select a valid LLM and Usecase option.")
        return {}

    
    user_message = st.chat_input("Enter your message:")

    if user_input.get("select_usecase") == "QEA Document Assistant":
        if not user_message and not st.session_state.get("analyze_clicked"):
            return
        if st.session_state.get("analyze_clicked"):
            print(user_message)

    
    if user_message:
        selected_model = user_input.get("select_llm")
        selected_temperature = user_input.get("select_temperature")
        print(f"Selected Model: {selected_model}")
        try:
            if selected_model == "OpenAI":
                print("OpenAI LLM selected")
                object_llm_object = OpenAILLM(user_controls_input=user_input)
                print("OpenAI LLM selected")
            if selected_model == "GROQ_AI":
                object_llm_object = GroqLLM(user_controls_input=user_input)
            if selected_model == "OLLAMA":
                print("Ollama LLM selected")
                object_llm_object = OllamaLLM(user_controls_input=user_input)
            if selected_model == "GOOGLE_AI":
                print("Google LLM selected")
                object_llm_object = GoogleLLM(user_controls_input=user_input)
            model = object_llm_object.get_llm_model()
            print(f"Model: {model}")

            if not model:
                st.error("Error : LLM Model could not be initialized")
                return
            
            usecase= user_input.get("select_usecase")
            print(usecase)
            enable_judge = user_input.get("enable_judge")
            test_case_format = user_input.get("tc_style")
            test_script_lang = user_input.get("script_lang")
            test_framework = user_input.get("framework")
            if not usecase:
                st.error("Error : No usecase selcted")
            
            #Graph Builder

            graph_builder_ = GraphBuilder(model,selected_temperature,enable_judge,test_case_format,test_script_lang,test_framework)

            try:
                graph = graph_builder_.setup_graph(usecase)
                session_id = user_input.get("session_id")

                DisplayResultStreamlit(usecase, graph, user_message, session_id,enable_judge).disply_result_on_ui()

                
                
            
            except Exception as e:
                st.error(f"Error : Graph setup failed - {e}")
                return 

        except Exception as e:
            return e 