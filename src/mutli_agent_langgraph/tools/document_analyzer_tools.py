import os
from src.mutli_agent_langgraph.tools.loader import load_pdf,load_docx,load_pptx,load_txt_csv,load_image_ocr,load_json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

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
    


def split_text_into_chunks(text, chunk_size=50, chunk_overlap=5):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def get_summary_chain(llm):
    prompt = PromptTemplate.from_template("""
    You are a document summarization assistant.
    Summarize the following content in a concise, clear, and structured format for a QA Engineer:

    {text}
    """)
    return LLMChain(llm=llm, prompt=prompt)


def summarize_text(text, llm):
    chunks = split_text_into_chunks(text)
    summary_chain = get_summary_chain(llm)
    all_summaries = [summary_chain.run(chunk) for chunk in chunks]
    return "\n\n".join(all_summaries)

def build_qa_engine(text, llm):
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

