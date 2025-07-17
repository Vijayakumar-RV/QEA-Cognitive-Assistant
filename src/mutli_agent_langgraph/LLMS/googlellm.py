import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os

class GoogleLLM:

    def __init__(self,user_controls_input):

        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        """
        Get the Google Gemini LLM model based on user input.
        
        Returns:
            ChatGenerative: An instance of the ChatGoogleGenerative model configured with the selected model and API key.
        """
        try:
            selected_model = self.user_controls_input["Selected_Google_Model"]
            print(f"selected_model : {selected_model}")
            selected_temperature = self.user_controls_input["select_temperature"]
            os.environ["GOOGLE_API_KEY"] = self.user_controls_input["GOOGLE_API_KEY"]
            google_key = self.user_controls_input["GOOGLE_API_KEY"]
            print(google_key)
            


            llm = ChatGoogleGenerativeAI(model=selected_model,
                                         temperature =selected_temperature)
            

        except Exception as e:
            raise ValueError(f"Error initializing Google LLM: {e}")
        
        return llm