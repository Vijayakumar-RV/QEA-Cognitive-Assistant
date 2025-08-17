from langchain_core.messages import HumanMessage, AIMessage
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation
from src.mutli_agent_langgraph.tools.document_analyzer_tools import summarize_text
from src.mutli_agent_langgraph.tools.document_analyzer_tools import build_qa_engine
from src.mutli_agent_langgraph.state.state import State

class DocumentAnalyzerNode:
    def __init__(self, llm, temperature):
        self.llm = llm
        self.temperature = temperature

    def process(self, state: State) -> dict:
        session_id = state.get("session_id")
        document_text = state.get("document_text")
        user_query = state.get("user_query")
        embedding_enabled = state.get("embedding_enabled", False)

        if not document_text:
            return {"error": "No document text found."}

        conv_mem = LangchainConversation(session_id)
        chat_history = conv_mem.get_conversation_memory_document().load_memory_variables({})["history"]
        working_context = chat_history.copy()

        if user_query:
            working_context.append(HumanMessage(content=user_query))
            if embedding_enabled:
                qa_chain = build_qa_engine(document_text, self.llm)
                response = qa_chain.run(user_query)
            else:
                prompt = f"""Here is the document content:\n\n{document_text}\n\nQuestion: {user_query}\nAnswer:"""
                response = self.llm.invoke([HumanMessage(content=prompt)]).content

            working_context.append(AIMessage(content=response))
            conv_mem.get_conversation_memory_document().save_context({"input": user_query}, {"output": response})
            return {"document_response": response, "messages": working_context}

        summary = summarize_text(document_text, self.llm)
        working_context.append(HumanMessage(content="Summarize this document"))
        working_context.append(AIMessage(content=summary))
        conv_mem.get_conversation_memory_document().save_context({"input": "Summarize this document"}, {"output": summary})

        return {"document_summary": summary, "messages": working_context}
    
