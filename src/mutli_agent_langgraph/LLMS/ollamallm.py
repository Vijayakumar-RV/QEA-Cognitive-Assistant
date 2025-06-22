import streamlit as st
from langchain_community.chat_models import ChatOllama

class OllamaLLM:
    def __init__(self, user_control_input):
        """
        Initializes the OllamaLLM instance with user controls input.
        
        Args:
            user_control_input (dict): A dictionary containing user inputs for LLM selection and API key.
        """
        self.user_control_input = user_control_input

    def get_llm_model(self):
        """
        Get the Ollama LLM model based on user input.
        
        Returns:
            OllamaChat: An instance of the OllamaChat model configured with the selected model and API key.
        """
        try:
            selected_model = self.user_control_input["Selected_Ollama_Model"]
            selected_temperature = self.user_control_input["select_temperature"]
              
            # Initialize the Ollama LLM model
            llm = ChatOllama(model=selected_model,temperature=selected_temperature)
            print(llm)

        except Exception as e:
            raise ValueError(f"Error initializing Ollama LLM: {e}")
        
        return llm