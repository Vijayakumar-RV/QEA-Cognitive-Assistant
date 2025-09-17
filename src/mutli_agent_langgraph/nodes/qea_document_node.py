from langchain_core.messages import HumanMessage, AIMessage
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation
from src.mutli_agent_langgraph.tools.document_analyzer_tools import summarize_text
from src.mutli_agent_langgraph.tools.document_analyzer_tools import build_qa_engine
from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import (
    log_params, log_metrics, log_text_artifact, log_json_artifact
)
from src.mutli_agent_langgraph.utils.timers import track_latency


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
        
           # --- RUN CONTEXT PARAMS ---
        try:
            log_params({
                "doc_node.session_id": session_id,
                "doc_node.embedding_enabled": bool(embedding_enabled),
                "doc_node.doc_chars": len(document_text or ""),
                "doc_node.llm_temperature": getattr(self, "temperature", None),
            })
            # Optional: keep a short preview as artifact
            log_text_artifact((document_text or "")[:4000], "doc_analyzer/input_document_preview.txt")
        except Exception:
            pass

        conv_mem = LangchainConversation(session_id)
        chat_history = conv_mem.get_conversation_memory_document().load_memory_variables({})["history"]
        working_context = chat_history.copy()

        if user_query:
            working_context.append(HumanMessage(content=user_query))

            try:
                log_text_artifact(user_query, "doc_analyzer/question.txt")
            except Exception:
                pass
        
            if embedding_enabled:
                with track_latency("doc_node.qa_build_latency_sec", log_metrics):
                    qa_chain = build_qa_engine(document_text, self.llm)

                with track_latency("doc_node.qa_latency_sec", log_metrics):
                    res = qa_chain.run(user_query)

                answer = res.get("result", "")
                sources = res.get("source_documents", []) or []
                cites = ", ".join(f"(chunk {d.metadata.get('chunk_id')})" for d in sources)
                response = f"{answer}\n\n**Sources**: {cites}" if sources else answer

                try:
                    log_text_artifact(answer, "doc_analyzer/answer.txt")
                    log_json_artifact(
                        [{"chunk_id": d.metadata.get("chunk_id"), "len": len(d.page_content or "")} for d in sources],
                        "doc_analyzer/citations.json"
                    )
                    log_metrics({
                        "doc_node.answer_chars": float(len(answer or "")),
                        "doc_node.citation_count": float(len(sources)),
                    })
                except Exception:
                    pass

            else:
                prompt = f"""Here is the document content:\n\n{document_text}\n\nQuestion: {user_query}\nAnswer:"""
                with track_latency("doc_node.direct_qa_latency_sec", log_metrics):
                    response = self.llm.invoke([HumanMessage(content=prompt)]).content
                try:
                    log_text_artifact(response or "", "doc_analyzer/answer_direct.txt")
                    log_metrics({"doc_node.answer_chars": float(len(response or ""))})
                except Exception:
                    pass

            working_context.append(AIMessage(content=response))
            conv_mem.get_conversation_memory_document().save_context({"input": user_query}, {"output": response})
            return {"document_response": response, "messages": working_context}
        
        with track_latency("doc_node.summarize_latency_sec", log_metrics):
            summary = summarize_text(document_text, self.llm)
        
        try:
            log_text_artifact(summary or "", "doc_analyzer/summary.txt")
            log_metrics({"doc_node.summary_chars": float(len(summary or ""))})
        except Exception:
            pass

        working_context.append(HumanMessage(content="Summarize this document"))
        working_context.append(AIMessage(content=summary))
        conv_mem.get_conversation_memory_document().save_context({"input": "Summarize this document"}, {"output": summary})

        return {"document_summary": summary, "messages": working_context}
    
