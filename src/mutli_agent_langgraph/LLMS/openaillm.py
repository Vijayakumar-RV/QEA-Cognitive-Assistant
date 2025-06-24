import streamlit as st
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

class OpenAILLM:

    def __init__(self,user_controls_input):
        self.user_controls_input = user_controls_input

    
    def get_llm_model(self):
        """
        Get the OpenAI LLM model based on user input.
        
        Returns:
            ChatOpenAI: An instance of the ChatOpenAI model configured with the selected model and API key.
        """
        
        try:
            
            load_dotenv()
            openai_key = os.getenv("OPENAI_API_KEY")
            selected_model = self.user_controls_input["Selected_OpenAI_Model"]

            if not openai_key:
                st.warning("⚠️ Please enter your OpenAI API Key.")
                return None
            
            # Initialize the OpenAI LLM model
            print(os.getenv("OPENAI_API_KEY"))
            print(os.getenv("OPENAI_API_BASE"))
            print(selected_model)
            llm = AzureChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                                azure_endpoint=os.getenv("OPENAI_API_BASE"),
                                openai_api_version=os.getenv("OPENAI_API_VERSION"),
                                deployment_name=selected_model,
                                temperature=1.0,streaming=True)

        except Exception as e:
            raise ValueError(f"Error initializing OpenAI LLM: {e}")
        
        return llm