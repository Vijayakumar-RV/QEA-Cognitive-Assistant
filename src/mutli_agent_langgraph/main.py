import streamlit as st
from src.mutli_agent_langgraph.ui.streamlit.load_ui import LoadStreamlitUI
from src.mutli_agent_langgraph.LLMS.groqllm import GroqLLM
from src.mutli_agent_langgraph.LLMS.ollamallm import OllamaLLM
from src.mutli_agent_langgraph.LLMS.openaillm import OpenAILLM
from src.mutli_agent_langgraph.graph.graph_builder import GraphBuilder
from src.mutli_agent_langgraph.ui.streamlit.display_results import DisplayResultStreamlit

def load_multi_agent_langgraph_ui():
    """
    Load the Streamlit UI for the Multi-Agent LangGraph application.
    This function initializes the LoadStreamlitUI class and loads the UI components
    based on the configuration settings defined in the Config class.
    Returns:
        dict: A dictionary containing user controls for the Streamlit UI.
    """
    # Initialize the LoadStreamlitUI class
    load_ui = LoadStreamlitUI()
    
    user_input = load_ui.load_streamlit_ui()

    if not user_input:
        st.warning("⚠️ Please select a valid LLM and Usecase option.")
        return {}
    
    user_message = st.text_input("Enter your message:", placeholder="Type your message here...",)

    if user_message:
        selected_model = user_input.get("select_llm")
        try:
            if selected_model == "GROQ_AI":
                object_llm_object = GroqLLM(user_controls_input=user_input)
            elif selected_model == "OPEN_AI":
                object_llm_object = OpenAILLM(user_controls_input=user_input)
            elif selected_model == "OLLAMA":
                object_llm_object = OllamaLLM(user_controls_input=user_input)
            model = object_llm_object.get_llm_model()

            if not model:
                st.error("Error : LLM Model could not be initialized")
                return
            
            usecase= user_input.get("select_usecase")
            print(usecase)

            if not usecase:
                st.error("Error : No usecase selcted")
            
            #Graph Builder

            graph_builder_ = GraphBuilder(model)

            try:
                graph = graph_builder_.setup_graph(usecase)
                DisplayResultStreamlit(usecase,graph,user_message).disply_result_on_ui()
            
            except Exception as e:
                st.error(f"Error : Graph setup failed - {e}")
                return 

        except Exception as e:
            return e 