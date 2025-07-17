from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.schema import message_to_dict, messages_from_dict

class LangchainConversation:
    def __init__(self, session_id : str, db_path = "C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\conversation_memory\\conversation_history.db"):

        self.session_id = session_id
        self.db_path = db_path
        
    def get_conversation_memory(self):
        """
        Get the conversation memory for the session.
        """
        print(f"Initializing conversation memory for session: {self.session_id} with db_path: {self.db_path}")
        try:
            chat_message_history = SQLChatMessageHistory(
                session_id=self.session_id,
                connection_string=f"sqlite:///{self.db_path}",
            )
            memory = ConversationBufferMemory(chat_memory=chat_message_history, return_messages=True,memory_key="history")
            return memory
        except Exception as e:
            print(f"Error initializing conversation memory: {e}")
            return None    
