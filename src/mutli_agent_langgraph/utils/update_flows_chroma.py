from langgraph.prebuilt import ToolNode
from langchain.schema import Document
import os
import json
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

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
            with open(os.path.join(flow_dir, filename), 'r') as f:
                flow = json.load(f)

            # Common metadata
            page_id = flow.get("page_id", "unknown")
            page_name = flow.get("page_name", "")
            description = flow.get("description", "")
            tags = flow.get("tags", [])
            tag_string = ", ".join(tags) if isinstance(tags, list) else str(tags)
            before_page = flow.get("before_page", [])
            after_page = flow.get("after_page", [])

            # Normalize to lists
            before_page = before_page if isinstance(before_page, list) else [before_page]
            after_page = after_page if isinstance(after_page, list) else [after_page]
           
           
            before_str = ", ".join(str(x) for x in before_page if x)
            after_str = ", ".join(str(x) for x in after_page if x)

            variants = flow.get("variants", [])
            raw_elements = flow.get("elements", [])
            elements = flatten_elements(raw_elements)


            if variants:
                for variant in variants:
                    variant_id = variant.get("variant_id", "default")
                    variant_desc = variant.get("description", "")
                    variant_elements = variant.get("elements", [])

                    steps = ""
                    for el in variant_elements:
                        label = el.get("label", "")
                        action = el.get("action_type", "")
                        locator = el.get("locator", {})
                        loc_type = locator.get("type", "unknown")
                        loc_val = locator.get("value", "")
                        steps += f"- {label} ({action}) | Locator: {loc_type} = {loc_val}\n"

                    content = (
                        f"Page ID: {page_id}\n"
                        f"Page Name: {page_name}\n"
                        f"Variant: {variant_id}\n"
                        f"Variant Description: {variant_desc}\n"
                        f"Before Page(s): {before_str}\n"
                        f"After Page(s): {after_str}\n"
                        f"Tags: {tag_string}\n"
                        f"---\nElements:\n{steps}"
                    )

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "page_id": page_id,
                            "variant_id": variant_id,
                            "page_name": page_name,
                            "tags": tag_string,
                            "source_file": filename
                        }
                    ))
            else:
                steps = ""
                for el in elements:
                    label = el.get("label", "")
                    action = el.get("action_type", "")
                    locator = el.get("locator", {})
                    loc_type = locator.get("type", "unknown")
                    loc_val = locator.get("value", "")
                    steps += f"- {label} ({action}) | Locator: {loc_type} = {loc_val}\n"

                content = (
                    f"Page ID: {page_id}\n"
                    f"Page Name: {page_name}\n"
                    f"Description: {description}\n"
                    f"Before Page(s): {before_str}\n"
                    f"After Page(s): {after_str}\n"
                    f"Tags: {tag_string}\n"
                    f"---\nElements:\n{steps}"
                )

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "page_id": page_id,
                        "variant_id": None,
                        "page_name": page_name,
                        "tags": tag_string,
                        "source_file": filename
                    }
                ))

    return documents


def update_save_flow_documents(DB_DIR):
    docs = flow_to_documents("C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\src\\mutli_agent_langgraph\\resources\\flows")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs,embedding=embedding_model,persist_directory=DB_DIR)
    vectorstore.persist()

    print(f"✅ Successfully embedded {len(docs)} flow chunks into ChromaDB.")

if __name__ == "__main__":
    update_save_flow_documents("C:\\Users\\Vijay's_Study_Nest\\MTech_Project\\QEA_Cognitive\\data_base\\chroma_memory\\ui_flow_memory\\ui_flow_v1")

    
