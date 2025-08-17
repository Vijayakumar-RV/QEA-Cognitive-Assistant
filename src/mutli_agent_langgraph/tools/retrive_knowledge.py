from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
import re
from typing import List, Optional, Tuple
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import log_params, log_metrics, log_json_artifact
import time
class Retrive:
    def __init__(self,
                 flow_DIR=r"C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\chroma_memory\\ui_flow_memory\\ui_flow_v2",
                 UR_DIR=r"C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\chroma_memory\\user_story_memory"):
        # single embedding instance used for both collections
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.flow_DIR = flow_DIR
        self.user_story_DIR = UR_DIR

        # light alias map to help exact page filters
        self.PAGE_ALIASES = {
            "checkout": "Checkout Page",
            "login": "Login/Register Page",
            "register": "Login/Register Page",
            "authentication": "Login/Register Page",
            "cart": "Cart Page",
            "basket": "Cart Page",
            "category": "Category Listing Page",
            "listing": "Category Listing Page",
            "search": "Search Results Page",
            "results": "Search Results Page",
            "specials": "Specials Page",
            "home": "Home Page",
            "product detail": "Product Detail Page",
            "pdp": "Product Detail Page",
        }

    # ---------- helpers ----------
    # >>> ADD: helper to stash retrieved text into state for the judge
    def _store_context_in_state(state: dict, docs, max_chars: int = 8000) -> dict:
        ctxs = []
        used = 0
        for d in docs or []:
            t = getattr(d, "page_content", "") or str(d)
            if not t:
                continue
            # limit total context size
            if used + len(t) > max_chars:
                t = t[: max(0, max_chars - used)]
            ctxs.append(t)
            used += len(t)
            if used >= max_chars:
                break
        state["retrieved_contexts"] = ctxs
        return state

    def _extract_flow_id(self, query: str) -> Optional[str]:
        # e.g., "flow CO" or "CO" (two uppercase letters)
        # keeps it conservative to avoid false positives
        m = re.search(r"\bflow\s*([A-Z]{2})\b", query or "", re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m2 = re.search(r"\b([A-Z]{2})\b", query or "")
        return m2.group(1).upper() if m2 else None

    def _page_from_query(self, query: str) -> Optional[str]:
        q = (query or "").lower()
        for key, page in self.PAGE_ALIASES.items():
            if key in q:
                return page
        return None

    def _normalize_story_id(self, query: str, pad: int = 2) -> Optional[Tuple[str, int]]:
        ID_PATTERNS = [
            re.compile(r"\bUS[-\s]?(\d{1,3})\b", re.IGNORECASE),
            re.compile(r"\buser\s*story\s*(\d{1,3})\b", re.IGNORECASE),
            re.compile(r"\bstory\s*(\d{1,3})\b", re.IGNORECASE),
        ]
        for pat in ID_PATTERNS:
            m = pat.search(query or "")
            if m:
                n = int(m.group(1))
                return f"US-{n:0{pad}d}".upper(), n
        return None

    def _extract_topic(self, query: str) -> Optional[str]:
        m = re.search(r"(?:related to|about|on|for)\s+([A-Za-z0-9 _-]+)", query or "", re.IGNORECASE)
        return m.group(1).strip().lower() if m else None
    
    def _approx_tokens(s: str) -> int:
        return max(1, len((s or "").split()))

    # ---------- UI Flow retrieval ----------
    def clean_flow_text(self,
                        query: str,
                        *,
                        top_k_exact: int = 3,
                        top_k_semantic: int = 12,
                        top_n_rerank: int = 6,
                        return_docs: bool = False) -> str | List[Document]:

        vectorstore = Chroma(persist_directory=self.flow_DIR, embedding_function=self.embedding_model)
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n_rerank)

        # 1) Flow ID filter
        flow_id = self._extract_flow_id(query)
        if flow_id:
            hits = vectorstore.similarity_search("", k=top_k_exact, filter={"flow_id": flow_id})
            if hits:
                return hits if return_docs else "\n\n".join(d.page_content for d in hits)

        # 2) Page name filter via alias
        page_name = self._page_from_query(query)
        if page_name:
            hits = vectorstore.similarity_search("", k=top_k_exact, filter={"page_name": page_name})
            if hits:
                return hits if return_docs else "\n\n".join(d.page_content for d in hits)

        # 3) Semantic fallback with MMR + cross-encoder re-rank
        base = vectorstore.as_retriever(search_type="mmr",
                                        search_kwargs={"k": top_k_semantic, "fetch_k": max(3*top_k_semantic, 36), "lambda_mult": 0.3})
        re_rank_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)

        log_params({
            "rag_source": "flows",
            "search_k": 8,
            "rerank_top_n": 5,
            "query": query
        })

        t0 = time.time()

        docs = re_rank_retriever.get_relevant_documents(query=query)
        latency = time.time() - t0
        log_metrics({"rag_latency_sec": latency, "rag_topk_count": float(len(docs))})

        # Save small snapshot of top-k (avoid dumping huge pages)
        payload = []
        for idx, d in enumerate(docs):
            md = getattr(d, "metadata", {}) or {}
            payload.append({
                "rank": idx + 1,
                "id": md.get("id", idx),
                "score": md.get("score"),
                "source": md.get("source"),
                "snippet": (d.page_content or "")[:1200]
            })
        log_json_artifact(payload, "rag/flows_topk.json")

        docs = docs[:top_n_rerank]

        return docs if return_docs else "\n\n".join(doc.page_content for doc in docs)

    # ---------- User Story retrieval ----------
    def clean_story_text(self,
                         query: str,
                         *,
                         top_k_exact: int = 3,
                         top_k_semantic: int = 12,
                         top_n_rerank: int = 6,
                         return_docs: bool = False) -> str | List[Document]:

        vectorstore = Chroma(persist_directory=self.user_story_DIR, embedding_function=self.embedding_model)
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n_rerank)

        # 1) ID-aware exact filters
        id_info = self._normalize_story_id(query)
        if id_info:
            canon_id, num = id_info
            hits = vectorstore.similarity_search("", k=top_k_exact, filter={"id": canon_id})
            if hits:
                return hits if return_docs else "\n\n".join(d.page_content for d in hits)
            hits = vectorstore.similarity_search("", k=top_k_exact, filter={"num": num})
            if hits:
                return hits if return_docs else "\n\n".join(d.page_content for d in hits)

        # 2) (Optional) topic tag filter if you added tags in metadata (safe to skip if not present)
        topic = self._extract_topic(query)
        if topic:
            tag_key = f"topic_{re.sub(r'[^a-z0-9]+','_', topic)}"
            try:
                tagged = vectorstore.similarity_search("", k=top_k_exact, filter={tag_key: True})
                if tagged:
                    return tagged if return_docs else "\n\n".join(d.page_content for d in tagged)
            except Exception:
                pass

        # 3) Semantic + cross-encoder rerank
        base = vectorstore.as_retriever(search_type="mmr",
                                        search_kwargs={"k": top_k_semantic, "fetch_k": max(3*top_k_semantic, 36), "lambda_mult": 0.3})
        re_rank_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)
        log_params({
            "rag_source": "flows",
            "search_k": 12,
            "rerank_top_n": 6,
            "query": query
        })

        t0 = time.time()
        docs = re_rank_retriever.get_relevant_documents(query=query)
        latency = time.time() - t0
        log_metrics({"rag_latency_sec": latency, "rag_topk_count": float(len(docs))})

        # Save small snapshot of top-k (avoid dumping huge pages)
        payload = []
        for idx, d in enumerate(docs):
            md = getattr(d, "metadata", {}) or {}
            payload.append({
                "rank": idx + 1,
                "id": md.get("id", idx),
                "score": md.get("score"),
                "source": md.get("source"),
                "snippet": (d.page_content or "")[:1200]
            })
        # was: log_json_artifact("rag/user_story_topk.json", payload)
        log_json_artifact(payload, "rag/user_story_topk.json")

        docs = docs[:top_n_rerank]

        return docs if return_docs else "\n\n".join(doc.page_content for doc in docs)
