from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

class Retrive:
    def __init__(self,
                 flow_DIR="C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\chroma_memory\\ui_flow_memory\\ui_flow_v1",
                 BR_DIR="src/mutli_agent_langgraph/resources/HuggingFace_Embeddings/BR_V1",
                 UR_DIR="C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\chroma_memory\\user_story_memory",):
        
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.flow_DIR = flow_DIR
        self.user_story_DIR = UR_DIR
        
  

    def clean_flow_text(self,query:str)->str:

        """
            Retrieves top-k flows based on the query,
            formats them as clean text, and returns them.
        """
        vectorstore = Chroma(
            persist_directory=self.flow_DIR,  # or your actual path
            embedding_function=self.embedding_model
            )
        base_Retriver = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":8}
        )

        compressor = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_n=5
        )

        re_rank_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_Retriver)
        
        docs = re_rank_retriever.get_relevant_documents(query=query)
        
        context = "\n\n".join(doc.page_content for doc in docs)

        print(context)

        return context
    

    def clean_story_text(self,query:str)->str:

        """
            Retrieves top-k user stories on the query,
            formats them as clean text, and returns them.
        """
        vectorstore = Chroma(
            persist_directory=self.user_story_DIR,  # or your actual path
            embedding_function=self.embedding_model
            )
        base_Retriver = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":8}
        )

        compressor = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_n=5
        )

        re_rank_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_Retriver)
        
        docs = re_rank_retriever.get_relevant_documents(query=query)
        
        context = "\n\n".join(doc.page_content for doc in docs)

        print(context)

        return context





    

    

