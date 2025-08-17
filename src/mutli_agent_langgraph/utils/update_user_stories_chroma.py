import os, json, re, uuid
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ID_RE = re.compile(r"\bUS-(\d{1,3})\b", re.IGNORECASE)

def _parse_num(story_id: str) -> int:
    """Extract numeric part from IDs like 'US-05' -> 5; returns -1 if not found."""
    m = ID_RE.search(story_id or "")
    return int(m.group(1)) if m else -1

def user_stories_to_documents(dirpath: str) -> List[Document]:
    documents: List[Document] = []

    # defensive: handle single file path too
    if os.path.isfile(dirpath) and dirpath.endswith(".json"):
        filepaths = [dirpath]
    else:
        filepaths = [
            os.path.join(dirpath, fn)
            for fn in os.listdir(dirpath)
            if fn.endswith(".json")
        ]

    for fp in filepaths:
        with open(fp, "r", encoding="utf-8") as f:
            try:
                user_stories = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping {fp}: JSON decode error: {e}")
                continue

        for story in user_stories:
            story_id = (story.get("id") or story.get("story_id") or "").strip()
            title = (story.get("title") or "").strip()
            user_story = (story.get("user_story") or "").strip()
            criteria = story.get("acceptance_criteria") or []

            # Extract numeric ID (e.g., US-05 -> 5). If missing, -1 (still indexable).
            num = _parse_num(story_id)

            criteria_text = "\n".join(f"- {ac}" for ac in criteria)

            page_content = (
                f"ID: {story_id}\n"
                f"Title: {title}\n"
                f"User Story: {user_story}\n"
                f"Acceptance Criteria:\n{criteria_text}"
            )

            # Stable doc id (helps avoid duplicates on re-ingest)
            stable_id = f"USERSTORY::{story_id}" if story_id else f"USERSTORY::{uuid.uuid4()}"

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        # 👇 keys your retriever expects
                        "id": story_id,      # e.g., "US-05"
                        "num": num,          # e.g., 5
                        "title": title,
                        "type": "user_story",
                        "source_file": os.path.basename(fp),
                        "doc_id": stable_id,
                    }
                )
            )

    return documents


def update_save_user_documents(DB_DIR: str, STORIES_DIR: str):
    # same embedding model your retriever uses
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build/attach to existing collection
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model
    )

    docs = user_stories_to_documents(STORIES_DIR)
    if not docs:
        print("⚠️ No user story documents found. Nothing to embed.")
        return

    # Avoid duplicates: upsert by doc_id if your Chroma wrapper supports ids.
    # LangChain's Chroma supports an `ids=` argument in add_documents.
    ids = [d.metadata.get("doc_id") for d in docs]
    vectorstore.add_documents(documents=docs, ids=ids)

    vectorstore.persist()
    print(f"✅ Successfully embedded/updated {len(docs)} user story docs into ChromaDB.")


if __name__ == "__main__":
    update_save_user_documents(
        DB_DIR=r"C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\chroma_memory\\user_story_memory",
        STORIES_DIR=r"C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\src\\mutli_agent_langgraph\\resources\\user_stories"
    )
