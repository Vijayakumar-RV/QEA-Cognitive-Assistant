import streamlit as st
from src.mutli_agent_langgraph.ui.streamlit.load_ui import LoadStreamlitUI


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