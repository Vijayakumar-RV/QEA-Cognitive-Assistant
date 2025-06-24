import streamlit as st
from langchain_community.chat_models import ChatOllama

class OllamaLLM:
    
    def __init__(self,user_controls_input):
        self.user_controls_input = user_controls_input


    def get_llm_model(self):
        """
        Get the Ollama LLM model based on user input.
        
        Returns:
            OllamaChat: An instance of the OllamaChat model configured with the selected model and API key.
        """
        print("Inside Ollama class")
        try:
            selected_model = self.user_controls_input["Selected_Ollama_Model"]
            selected_temperature = self.user_controls_input["select_temperature"]
            
            if not selected_model:
                st.warning("⚠️ Please select an Ollama model.")
                return None

            llm = ChatOllama(model=selected_model,temperature=selected_temperature,disable_streaming=False)
            
        except Exception as e:
            raise ValueError(f"Error initializing Ollama LLM: {e}")
        
        return llm