# QEA Cognitive Assistant

## 📌 Overview

The **QEA Cognitive Assistant** is a multi-agent, LangGraph-powered AI system designed to assist in **Quality Engineering and Assurance (QEA)**.
It enables **automatic test case generation, automation script creation, document analysis, knowledge retrieval, and research assistance** for web and robotics-driven test environments.

This project is part of an **M.Tech dissertation** and is built with a modular, extensible architecture using **Streamlit UI + LangGraph orchestration + ChromaDB retrieval**.

---

## 🚀 Features Implemented So Far

### 🔹 Core QEA Assistant

* Conversational chatbot that generates **manual test cases** and **automation test scripts** (Selenium, Cypress, Playwright, Java, Python, JS).
* **Retriever tools** to fetch knowledge from:

  * **UI Flows (JSON schemas)**
  * **User Stories**
* **Context-aware generation**: Preserves flow progression, before/after pages, and locators.
* **Persistent conversation memory** using SQLite (LangChain SQLChatMessageHistory).

### 🔹 Multi-Agent Orchestration

* Built on **LangGraph StateGraph** with nodes for:

  * **Chatbot (QEAAssistantChatbot)**
  * **Retriever Tools (flows & user stories)**
  * **Document Analyzer**
  * **Research Agent**
* Conditional edges ensure the assistant decides **when to use retrieval, analysis, or research**.

### 🔹 Document Analyzer

* Supports **.pdf, .docx, .pptx, .csv, .txt, .jpeg** and OCR for screenshots.
* Provides summaries and contextual answers from uploaded docs.
* Integrated as a **separate LangGraph use case** with its own tools and node.

### 🔹 Research Agent

* Queries the **web in real time** to fetch the latest information.
* Summarizes and contextualizes research results for QEA scenarios.
* Integrated as a **LangGraph branch** alongside other agents.

### 🔹 LLM Integrations

Supports multiple LLM backends via modular wrappers:

* **OpenAI (Azure)**
* **Groq**
* **Google (Gemini)**
* **Ollama (local models)**

### 🔹 Streamlit UI

* Unified **frontend with dropdowns** for:

  * LLM selection
  * Model name, temperature, API keys
  * Use case (QEA Assistant, Document Analyzer, Research Agent)
* Chat-style interface with **real-time streaming responses**.

### 🔹 Knowledge Base

* **UI flow JSONs** and **User story JSONs** embedded in ChromaDB.
* Re-ranking with **cross-encoder** for high-quality retrieval.
* Rich context: `before_page`, `after_page`, variants, and locators.

### 🔹 Tracking & Logging

* **MLflow integration** for:

  * Experiment tracking
  * Evaluation metrics
  * Judge-based scoring for test cases & scripts.

### 🔹 Guardrails (LlamaGuard3)

* Detects **harmful or unsafe user inputs** before processing.
* Enforces **structured JSON output** for assistant responses.

---

## 🏗️ Project Architecture

```mermaid
graph TD
  A[Streamlit UI] --> B[LangGraph StateGraph]
  B --> C[QEA Assistant Node]
  B --> D[Retriever Tools]
  B --> E[Document Analyzer Node]
  B --> F[Research Agent Node]
  C --> G[LLM Wrappers: OpenAI, Groq, Google, Ollama]
  D --> H[ChromaDB: UI Flows + User Stories]
  E --> I[Doc Loader + OCR + Parsers]
  F --> J[Web Search + Summarization]
  B --> K[Memory (SQLite)]
  B --> L[MLflow Tracking]
  B --> M[LlamaGuard Guardrails]
```

---

## 📂 Directory Structure

```
src/multi_agent_langgraph/
│── graph/                # GraphBuilder & StateGraph setup
│── llm/                  # LLM wrappers (OpenAI, Groq, Google, Ollama)
│── memory/               # Conversation memory (SQLite backend)
│── nodes/                # Assistant chatbot, document analyzer, research agent
│── resources/            # UI flow & user story JSONs
│── state/                # State definition (LangGraph state class)
│── tools/                # Retrieval tools & document analyzer tools
│── ui/                   # Streamlit UI components
│── utils/                # Utilities (embedding, flow_to_documents, etc.)
│── main.py               # Entry point for Streamlit app
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/qea-cognitive-assistant.git
cd qea-cognitive-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv cogvenv
source cogvenv/bin/activate   # (Linux/Mac)
cogvenv\Scripts\activate      # (Windows)
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file for API keys:

```
OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

### 5. Run Streamlit App

```bash
streamlit run src/multi_agent_langgraph/main.py
```

---

## 🖥️ Usage

* **Select LLM** (OpenAI, Groq, Google, Ollama)
* **Pick Use Case**:

  * **QEA Assistant** → Test case & script generation
  * **Document Analyzer** → Upload and analyze docs
  * **Research Agent** → Web search + summarization
* **Interact** with the assistant in real time.

---

## 📊 Evaluation

* **Guardrails** (LlamaGuard3) ensure safe & structured outputs.
* **Judge-based evaluation** for generated test cases.
* **MLflow logging** for metrics (quality, correctness, coverage).
* **Comparison across multiple LLMs** for benchmarking.

---

## 🔮 Roadmap

* [ ] **JIRA & CI/CD Integration** for real-world adoption
* [ ] **Robotics Test Automation Dashboard** (Physical vs Digital test case classification)
* [ ] **Enhanced Visualization** of metrics and flows in Streamlit

---
