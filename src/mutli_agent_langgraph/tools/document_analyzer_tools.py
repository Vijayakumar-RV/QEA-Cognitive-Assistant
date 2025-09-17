import os
from src.mutli_agent_langgraph.tools.loader import load_pdf,load_docx,load_pptx,load_txt_csv,load_image_ocr,load_json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import log_params, log_metrics, log_text_artifact
from src.mutli_agent_langgraph.utils.timers import track_latency
import hashlib
from pathlib import Path

def parse_document(file):
    filename = file.name
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return load_pdf(file)
    elif ext == ".docx":
        return load_docx(file)
    elif ext == ".pptx":
        return load_pptx(file)
    elif ext in [".txt", ".csv"]:
        return load_txt_csv(file)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return load_image_ocr(file)
    elif ext == ".json":
        return load_json(file)
    else:
        return "Unsupported file format", ""
    
CACHE_DIR = Path("artifacts/doc_cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
_EMB = None
def _embeddings():
    global _EMB
    if _EMB is None:
        _EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try: log_params({"doc_node.embedding_model": "all-MiniLM-L6-v2"})
        except Exception: pass
    return _EMB

def _doc_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]

def split_text_into_chunks(text, chunk_size=50, chunk_overlap=5):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
        is_separator_regex=False

    )

    parts = splitter.split_text(text or "")
    docs = [Document(page_content=t, metadata={"chunk_id": i+1}) for i, t in enumerate(parts)]
    # MLflow stats
    try:
        avg = (sum(len(d.page_content) for d in docs) / max(len(docs),1))
        log_metrics({
            "doc_node.chunks": float(len(docs)),
            "doc_node.chunk_size_avg": float(avg),
            "doc_node.chunk_size_param": float(chunk_size),
            "doc_node.chunk_overlap_param": float(chunk_overlap),
        })
    except Exception:
        pass
    return docs

MAP_PROMPT = PromptTemplate.from_template("""
You are a precise analyst. Summarize the following passage with bullet points, keeping facts and numbers.
Passage:
{text}
""")

REDUCE_PROMPT = PromptTemplate.from_template("""
Combine these partial summaries into a concise, well-structured brief for a QA Engineer.
Keep sections: Overview, Key Entities, Flows/Steps, Data/Constraints, Risks.
Summaries:
{text}
""")


def summarize_text(text, llm):
    chunks = split_text_into_chunks(text)
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                         map_prompt=MAP_PROMPT, 
                                         combine_prompt=REDUCE_PROMPT, 
                                         return_intermediate_steps=False,
                                         verbose=False)
    with track_latency("doc_node.summarize_chain_latency_sec", log_metrics):
        summary =  summary_chain.run(chunks)
    try:
         log_metrics({"doc_node.summary_tokens_approx": float(len(summary or "")/4.0)})
    except Exception:
        pass
    return summary

def build_qa_engine(text: str, llm, k: int = 4):
    h = _doc_hash(text)
    idx_dir = CACHE_DIR / f"faiss_{h}"
    cache_hit = idx_dir.exists()

    if cache_hit:
        vectordb = FAISS.load_local(str(idx_dir), _embeddings(), allow_dangerous_deserialization=True)
    else:
        docs = split_text_into_chunks(text)
        vectordb = FAISS.from_documents(docs, embedding=_embeddings())
        vectordb.save_local(str(idx_dir))

    try:
        log_params({"doc_node.faiss_cache_dir": str(idx_dir)})
        log_metrics({"doc_node.faiss_cache_hit": 1.0 if cache_hit else 0.0})
    except Exception:
        pass

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    try:
        log_metrics({"doc_node.retriever_k": float(k)})
    except Exception:
        pass

    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    QA_PROMPT = PromptTemplate.from_template("""
    Answer ONLY from the context. If unknown, say "Not found in the provided document."
    Cite chunk ids like (chunk 3, 5).

    Question: {question}

    Context:
    {context}
    """)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return qa

