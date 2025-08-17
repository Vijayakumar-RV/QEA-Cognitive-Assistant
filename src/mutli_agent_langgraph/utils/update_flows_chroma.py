from langgraph.prebuilt import ToolNode
from langchain.schema import Document
import os
import json
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
def flatten_elements(elements, parent_section=None):
    flat_elements = []
    for item in elements:
        if "elements" in item and isinstance(item["elements"], list):
            # It's a nested section
            section_name = item.get("section") or item.get("title") or parent_section
            flat_elements.extend(flatten_elements(item["elements"], section_name))
        else:
            # It's a UI element
            element_copy = item.copy()
            if parent_section:
                element_copy["section"] = parent_section
            flat_elements.append(element_copy)
    return flat_elements


def flow_to_documents(flow_dir):
    documents = []

    for filename in os.listdir(flow_dir):
        if filename.endswith(".json"):
            with open(os.path.join(flow_dir, filename), 'r', encoding='utf-8') as f:
                flow = json.load(f)

            flow_id = flow.get("flow_id", "")
            page_name = flow.get("page", "")
            title = flow.get("title", "")
            description = flow.get("description", "")

            # Steps (narrative)
            steps_data = flow.get("steps", [])
            steps_str = "\n".join(
                f"{i+1}. {s.get('user_action','')} → {s.get('expected_behavior','')}"
                for i, s in enumerate(steps_data)
            )

            # Elements (locator info)
            raw_elements = flow.get("elements", [])
            elements = flatten_elements(raw_elements)
            elements_str = ""
            for el in elements:
                label = el.get("label", "")
                action = el.get("action_type", "")
                locator = el.get("locator", {})
                loc_type = locator.get("type", "unknown")
                loc_val = locator.get("value", "")
                section = el.get("section", "")
                elements_str += f"- {label} ({action}) [Section: {section}] | Locator: {loc_type} = {loc_val}\n"

            # Variants (if any)
            variants = flow.get("variants", [])
            if variants:
                for variant in variants:
                    variant_id = variant.get("variant_id", "default")
                    variant_desc = variant.get("description", "")

                    variant_elements = variant.get("elements", [])
                    variant_elements_str = ""
                    for el in variant_elements:
                        label = el.get("label", "")
                        action = el.get("action_type", "")
                        locator = el.get("locator", {})
                        loc_type = locator.get("type", "unknown")
                        loc_val = locator.get("value", "")
                        variant_elements_str += f"- {label} ({action}) | Locator: {loc_type} = {loc_val}\n"

                    content = (
                        f"Flow ID: {flow_id}\n"
                        f"Page: {page_name}\n"
                        f"Title: {title}\n"
                        f"Description: {description}\n"
                        f"Variant: {variant_id} - {variant_desc}\n"
                        f"---\nSteps (Narrative):\n{steps_str}\n"
                        f"---\nElements (Locators):\n{variant_elements_str or elements_str}"
                    )

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "flow_id": flow_id,
                            "page_name": page_name,
                            "variant_id": variant_id,
                            "source_file": filename
                        }
                    ))
            else:
                content = (
                    f"Flow ID: {flow_id}\n"
                    f"Page: {page_name}\n"
                    f"Title: {title}\n"
                    f"Description: {description}\n"
                    f"---\nSteps (Narrative):\n{steps_str}\n"
                    f"---\nElements (Locators):\n{elements_str}"
                )

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "flow_id": flow_id,
                        "page_name": page_name,
                        "variant_id": None,
                        "source_file": filename
                    }
                ))

    return documents

def update_save_flow_documents(DB_DIR: str, FLOW_DIR: str):
    """
    Reads all flow JSON files from FLOW_DIR, converts them to Document objects
    with both narrative steps & locator info, and saves them into ChromaDB at DB_DIR.
    """
    # Generate documents from the flows
    docs = flow_to_documents(FLOW_DIR)
    if not docs:
        print("⚠️ No UI flow documents found. Nothing to embed.")
        return

    # Embedding model — must match the one you use at retrieval time
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create a new Chroma collection or overwrite existing
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )

    # Save to disk
    vectorstore.persist()
    print(f"✅ Successfully embedded {len(docs)} UI flow chunks into ChromaDB.")

# Example usage:
if __name__ == "__main__":
    update_save_flow_documents(
        DB_DIR=r"C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\chroma_memory\\ui_flow_memory\\ui_flow_v2",
        FLOW_DIR=r"C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\src\\mutli_agent_langgraph\\resources\\flows"
    )
    
