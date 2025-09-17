import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,BaseMessage
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation
import time
from src.mutli_agent_langgraph.tools.document_analyzer_tools import parse_document, build_qa_engine,summarize_text
import time
import json
import subprocess
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import (
    run_mlflow_run, log_params, log_metrics, log_text_artifact, log_json_artifact
)

class DisplayResultStreamlit:
    def __init__(self,usecase,graph,user_message,session_id,enable_judge):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.session_id = session_id
        self.enable_judge = enable_judge


    def _render_message(self, msg: BaseMessage):
        """Render any Chat / Tool message in Streamlit chat UI."""
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)

        # show assistant replies **only** if they have no tool_calls
        elif msg.type == "ai" and not getattr(msg, "tool_calls", None):
            with st.chat_message("assistant"):
                st.markdown(msg.content,unsafe_allow_html=True)


    def disply_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message
        session_id = self.session_id

        # ===================== BEGIN PATCH: QEA_Assistant with Judge & Exports =====================
        if usecase == "QEA_Assistant":
            # --- controls & helpers ---

            enable_judge = bool(self.enable_judge)

            # Optional: allow comma-separated hints in UI and store under this key
            eval_hints_csv = " " # e.g., "Checkout, Guest Checkout, billing, Confirm"
            eval_gold = {
                "must_include": [h.strip() for h in str(eval_hints_csv).split(",") if str(h).strip()]
            } if enable_judge and eval_hints_csv else {}

            conversation_memory = LangchainConversation(session_id=session_id)
            history_msgs = conversation_memory.get_conversation_memory().load_memory_variables({})["history"]
            for old_msg in history_msgs:
                self._render_message(old_msg)

            if not user_message:
                return

            with st.chat_message("user"):
                st.markdown(user_message)

            assistant_block = st.chat_message("assistant")
            stream_area = assistant_block.empty()

            # Build the graph input state. Pass usecase; pass eval_gold only when judge is enabled.
            state_input = {
                "messages": [HumanMessage(content=user_message)],
                "session_id": session_id,
                "usecase": usecase,
            }
            if enable_judge:
                state_input["eval_gold"] = eval_gold

            # Build preview
            hist_preview = "\n".join(f"{m.type}: {getattr(m,'content','')}" for m in history_msgs[-4:])
            prompt_preview = f"{hist_preview}\nuser: {user_message}\nassistant:"

            run_tags = {"usecase": usecase, "session_id": session_id}
            run_params = {"ui_component": "display_result_on_ui", "messages_len_before": len(history_msgs),
                        "enable_judge": enable_judge}

            full_answer_chunks = []
            token_estimate = 0
            t0 = time.time()
            last_state = None  # <-- capture final streamed state so we can read evaluation

            # >>> keep the run open across streaming + final logging <<<
            with run_mlflow_run(run_name=f"{usecase}_{session_id}", tags=run_tags, params=run_params):
                log_text_artifact(prompt_preview[:8000], "prompts/ui_prompt_preview.txt")
                # uiconfig snapshot
                try:
                    log_text_artifact(open("ui/streamlit/uiconfig.ini","r",encoding="utf-8").read(),
                                    "manifest/uiconfig.ini")
                except Exception:
                    pass

                # git commit
                try:
                    commit = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
                    log_text_artifact(commit, "manifest/git_commit.txt")
                except Exception:
                    pass

                # pip freeze
                try:
                    freeze = subprocess.check_output(["pip","freeze"], text=True)
                    log_text_artifact(freeze, "manifest/requirements_freeze.txt")
                except Exception:
                    pass

                # graph structure (adjust if you add judge in builder)
                graph_info = {
                    "nodes": ["planner","retriver","qeacognitive"] + (["judge"] if enable_judge else []),
                    "edges": [("START","planner"),
                            ("planner","retriver"),
                            ("retriver","qeacognitive")] + ([("qeacognitive","judge"),("judge","END")] if enable_judge else [("qeacognitive","END")])
                }
                log_json_artifact(graph_info, "manifest/graph.json")

                try:
                    with st.spinner("🤖 Generating response..."):
                        for event in graph.stream(state_input):
                            for value in event.values():
                             
                                last_state = value

                                if "messages" not in value:
                                    continue
                                last_msg = value["messages"][-1]

                                # skip tool call placeholder chunks
                                if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                                    continue

                                if isinstance(last_msg, AIMessage):
                                    stream_area.markdown(last_msg.content,unsafe_allow_html=True)
                                    #stream_area.code(last_msg.content, language="markdown",unsafe_allow_html=True)
                                    text = last_msg.content or ""
                                    full_answer_chunks.append(text)
                                    token_estimate += max(1, len(text.split()))
                             

                except Exception as e:
                    log_text_artifact(str(e), "errors/ui_stream_exception.txt")
                    log_params({"ui_last_error_type": e.__class__.__name__})
                    raise
                
                # finalize logs within the same run
                total_latency = time.time() - t0
                full_answer = "\n".join(full_answer_chunks).strip()
                log_text_artifact(full_answer[:200_000], "outputs/ui_streamed_answer.txt")
                log_metrics({
                    "ui_stream_latency_sec": float(total_latency),
                    "ui_tokens_estimate": float(token_estimate),
                    "messages_len_after": float(len(history_msgs) + 1),
                })
                from datetime import datetime

                def _render_timeline(prompt, retrieved, answer):
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    return f"""<html><body>
                    <h2>QEA Session Timeline</h2>
                    <p><b>Timestamp:</b> {ts}</p>
                    <h3>Prompt</h3><pre>{prompt}</pre>
                    <h3>Retrieved Context (preview)</h3><pre>{(retrieved or '')[:4000]}</pre>
                    <h3>Final Answer</h3><pre>{answer}</pre>
                    </body></html>"""
                try:
                    # show preview of what retriever stored (if any)
                    retrieved_preview = ""
                    if last_state and isinstance(last_state, dict):
                        retrieved_preview = "\n\n---\n".join(last_state.get("retrived_content", []) or [])
                    timeline_html = _render_timeline(prompt_preview, retrieved_preview, full_answer)
                    log_text_artifact(timeline_html, "reports/session_timeline.html")
                except Exception:
                    pass


        elif usecase=="QEA Research Assistant":
            initial_state = {"messages":[user_message]}
            res = graph.invoke(initial_state)
            for message in res["messages"]:
                if type(message) == HumanMessage:
                    with st.chat_message("user"):
                        st.write(message.content)
                elif type(message) == ToolMessage:
                    with st.chat_message("ai"):
                        st.write("Tool call Started")
                        st.write(message.content)
                        st.write("Tool call Ended")

                elif type(message) == AIMessage and message.content:
                    with st.chat_message("assistant"):
                        st.write(message.content)

        
        elif usecase == "QEA Document Assistant":
            run_tags = {"usecase": usecase, "session_id": session_id}
            run_params = {"ui_component": "display_result_on_ui"}

            with run_mlflow_run(run_name=f"{usecase}_{session_id}", tags=run_tags, params=run_params):
                log_params({"session_id": session_id})
            state_input = {
            "session_id": session_id,
            "document_text": st.session_state.get("document_text"),
            "user_query": user_message,
            "embedding_enabled": st.session_state.get("embedded_store") is not None
        }

            result = graph.invoke(state_input)

            st.subheader("🔎 AI Response")
            if "error" in result:
                st.error(result["error"])
            elif "document_summary" in result:
                st.success("📄 Summary:")
                st.write(result["document_summary"])
            elif "document_response" in result:
                st.success("💬 Answer:")
                st.write(result["document_response"])

                    # Show past conversation
            if "messages" in result and len(result["messages"]) > 0:
                for msg in result["messages"]:
                    if isinstance(msg, HumanMessage):
                        with st.chat_message("user"):
                            st.markdown(f"🧑‍💻 **User:**\n{msg.content}")
                    elif isinstance(msg, AIMessage):
                        with st.chat_message("assistant"):
                            st.markdown(f"🤖 **Assistant**\n{msg.content}")
