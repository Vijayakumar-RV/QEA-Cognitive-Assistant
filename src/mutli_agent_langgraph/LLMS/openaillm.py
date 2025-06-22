import streamlit as st
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

class OpenAILLM:

    def __init__(self, user_control_input):
        """
        Initializes the OpenAILLM instance with user controls input.
        
        Args:
            user_control_input (dict): A dictionary containing user inputs for LLM selection and API key.
        """
        load_dotenv()
        self.user_control_input = user_control_input

    def get_llm_model(self):
        """
        Get the OpenAI LLM model based on user input.
        
        Returns:
            ChatOpenAI: An instance of the ChatOpenAI model configured with the selected model and API key.
        """
        try:
            openai_key = self.user_control_input["OPENAI_API_KEY"]
            selected_model = self.user_control_input["Selected_OpenAI_Model"]
            #selected_temperature = self.user_control_input["select_temperature"]

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
                                temperature=1.0)

        except Exception as e:
            raise ValueError(f"Error initializing OpenAI LLM: {e}")
        
        return llm