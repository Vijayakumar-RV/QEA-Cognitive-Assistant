from langchain.schema import Document
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

def user_stories_to_documents(filepath: str):
    documents = []

    for filename in os.listdir(filepath):
        if filename.endswith(".json"):
            with open((os.path.join(filepath,filename)), "r") as f:
                user_stories = json.load(f)

            for story in user_stories:
                story_id = story.get("id", "")
                title = story.get("title", "")
                user_story = story.get("user_story", "")
                criteria = story.get("acceptance_criteria", [])

                criteria_text = "\n".join(f"- {ac}" for ac in criteria)

                page_content = (
                    f"User Story ID: {story_id}\n"
                    f"Title: {title}\n"
                    f"Story: {user_story}\n"
                    f"Acceptance Criteria:\n{criteria_text}"
                )

                documents.append(Document(
                    page_content=page_content,
                    metadata={
                        "story_id": story_id,
                        "title": title,
                        "type": "user_story"
                    }
                ))

            return documents


def update_save_user_documents(DB_DIR):
    docs = user_stories_to_documents("C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\src\\mutli_agent_langgraph\\resources\\user_stories")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs,embedding=embedding_model,persist_directory=DB_DIR)
    vectorstore.persist()

    print(f"✅ Successfully embedded {len(docs)} flow chunks into ChromaDB.")

if __name__ == "__main__":
    update_save_user_documents("C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\chroma_memory\\user_story_memory")