import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    
    def __init__(self,user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        """
        Get the Groq LLM model based on user input.
        Returns:
            ChatGroq: An instance of the ChatGroq model configured with the selected model and API key.
        """
        try:
        
            groq_api_key = self.user_controls_input["GROQ_API_KEY"]
            print(groq_api_key)
            selected_model = self.user_controls_input["Selected_Groq_Model"]
            print(selected_model)
            selected_temperature = self.user_controls_input["select_temperature"]


            if not groq_api_key:
                st.warning("⚠️ Please enter your Groq API Key.")
                return None
            
            # Initialize the Groq LLM model
            llm = ChatGroq(model=selected_model, api_key=groq_api_key,temperature=selected_temperature)
            print(llm)
        except Exception as e:
            raise ValueError(f"Error initializing Groq LLM: {e}")
            
        return llm