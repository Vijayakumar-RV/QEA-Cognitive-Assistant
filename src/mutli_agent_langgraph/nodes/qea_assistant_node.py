from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from src.mutli_agent_langgraph.agents import testcase_agent_, testscript_agent, userstory_agent, save_execute_agent
import time
import hashlib
from src.mutli_agent_langgraph.utils.llm_inspect import infer_llm_metadata
from src.mutli_agent_langgraph.utils.session_store import save_session_rows,load_session_rows
from src.mutli_agent_langgraph.utils.tracking.mlflow_utils import (
    log_params, log_metrics, log_text_artifact, log_json_artifact
)
from src.mutli_agent_langgraph.utils.session_store import save_session_script_blob,load_session_script_blob

# Llama Guard 3 (output moderation)
from src.mutli_agent_langgraph.guardrails.llamaguard3 import LlamaGuard3


class QEAAssistantChatbot:

    def __init__(self, model, temperature, test_case_format: str = "", test_script_lang: str = "", test_framework: str = ""):
        self.llm = model
        self.temperature = temperature
        self.moderator = LlamaGuard3()
        self.test_case_format = test_case_format
        self.test_script_lang = test_script_lang
        self.test_framework = test_framework

    def chit_irrelevant(self):
        pass

    def _needs_context(self, plan_generation) -> bool:
        if not plan_generation:
            return False
        wanted = set(x.lower() for x in plan_generation or [])
        return bool(wanted.intersection({"testcase", "testscript", "userstory"}))

    def process(self, state: State) -> State:
        """Process the input state and generate a QEA chatbot response."""
        session_id = state["session_id"]
        user_input = state["messages"][-1].content

        safety = state.get("safety", {}) or {}
        if safety.get("blocked_by") == "llamaguard_input":
            refusal = "I can’t assist with that request."
            state["messages"].append(AIMessage(content=refusal))
            # (optional) save to memory
            try:
                LangchainConversation(session_id=session_id).get_conversation_memory()\
                    .save_context({"input": user_input}, {"output": refusal})
            except Exception:
                pass
            return state

        conversation_memory = LangchainConversation(session_id=session_id)
        memory = conversation_memory.get_conversation_memory()
        chat_history = memory.load_memory_variables({})['history']

        try:
            history = "\n".join(f"{message.type} : {message.content}" for message in chat_history)
        except Exception:
            history = ""
        prompt = f"{history}\nuser: {user_input}\nassistant:"

        prompt_hash = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:12]
        log_params({"prompt_hash": prompt_hash})
        log_text_artifact((prompt or "")[:8000], f"prompts/{prompt_hash}.txt")
        print(f"Prompt hash: {prompt_hash}")

        # Planner outputs
        plan_obj        = state.get("plan", None)
        plan_generation = getattr(plan_obj, "generation", None)
        retriver        = state.get("retrived_content", "")
        plan_irrelevant = getattr(plan_obj, "irrelevant", None)
        plan_chitchat   = getattr(plan_obj, "chitchat", None)
        save_tc         = getattr(plan_obj, "save_testcases", None)
        save_ts         = getattr(plan_obj, "save_execute_testscripts", None)

        # MLflow params/artifacts
        try:
            vendor, model_name, temperature, extras = infer_llm_metadata(self.llm)
            log_params({
                "node": "QEAAssistantChatbot",
                "llm_vendor": vendor,
                "llm_model": model_name,
                "llm_temperature": temperature,
                "plan_generation": str(plan_generation),
                "has_irrelevant": bool(plan_irrelevant),
                "has_chitchat": bool(plan_chitchat),
                "save_tc": str(save_tc),
                "save_ts": str(save_ts),
                "history_len": len(chat_history),
            })
            if extras.get("openai_api_type"): log_params({"openai_api_type": extras["openai_api_type"]})
            if extras.get("azure_endpoint"):   log_params({"azure_endpoint": extras["azure_endpoint"]})
            log_text_artifact((prompt or "")[:8000], "prompts/node_prompt_preview.txt")
            if retriver:
                preview = retriver if isinstance(retriver, str) else str(retriver)
                log_text_artifact(preview[:12000], "rag/node_last_retrieved.txt")
        except Exception:
            pass

        # RAG-only enforcement
        try:
            require_context = self._needs_context(plan_generation)
            have_context = bool((retriver or "").strip())
            log_metrics({
                "rag_require_context": 1.0 if require_context else 0.0,
                "rag_have_context_in_qea_node": 1.0 if have_context else 0.0
            })
            if require_context and not have_context:
                refusal = "I don’t have enough verified project context to answer. Please sync the flow or upload the page schema."
                state["messages"].append(AIMessage(content=refusal))
                log_metrics({"rag_only_refusal": 1.0})
                try: memory.save_context({"input": user_input}, {"output": refusal})
                except Exception: pass
                return state
        except Exception:
            pass

        t_node0 = time.time()
        branch_latency = {}

        def _time_call(label, fn, *args, **kwargs):
            t0 = time.time()
            out = fn(*args, **kwargs)
            branch_latency[label] = time.time() - t0
            return out

        try:
            if plan_generation and "testcase" in plan_generation:
                updated_state = _time_call("agent_testcase",
                    testcase_agent_.testcase, retriver=retriver, user_message=prompt, model=self.llm, state=state, test_case_format=self.test_case_format)
                state.update(updated_state)
                rows_now = state.get("test_rows", [])
                if rows_now:
                    save_session_rows(session_id, rows_now)


            elif plan_generation and "testscript" in plan_generation:
                updated_state = _time_call("agent_testscript",
                    testscript_agent.testscript, session_id=session_id, retriver=retriver, user_message=prompt, model=self.llm, state=state, test_script_lang=self.test_script_lang, test_framework=self.test_framework)
                state.update(updated_state)
                scripts = state.get("testscript", {})
                if scripts:
                    save_session_script_blob(session_id, scripts)

            elif plan_generation and "userstory" in plan_generation:
                _time_call("agent_userstory",
                    userstory_agent.userstory, retriver=retriver, user_message=prompt, model=self.llm, state=state)

            elif plan_irrelevant:
                _time_call("Irrelevant content", self.chit_irrelevant)
                state["messages"].append(AIMessage(content=plan_irrelevant))

            elif plan_chitchat:
                _time_call("Chitchat", self.chit_irrelevant)
                state["messages"].append(AIMessage(content=plan_chitchat))

            elif save_ts:
                test_rows = load_session_rows(session_id)
                filename = test_rows[0].get("Tags").split(",")[0] if test_rows else "last_testcase"
                
                if "save" in (save_ts or []) or "both" in (save_ts or []):
                    script_content = load_session_script_blob(session_id)
                    filename_return = _time_call("agent_save_script", save_execute_agent.save_testscript_execute, status="save", script_content=script_content, filename=filename, test_script_lang=self.test_script_lang)
                    state["messages"].append(AIMessage(content=f"✅ Saved Test Script: {filename_return}"))
                if "execute" in (save_ts or []) or "both" in (save_ts or []):
                    filename_return = _time_call("agent_execute_script", save_execute_agent.save_testscript_execute, status="run", script_content=script_content, filename=filename, test_script_lang=self.test_script_lang)
                    state["messages"].append(AIMessage(content=f"✅ Saved Test Script: {filename_return}"))
            # Save test cases
            try:
                print(f"Generated test cases in test: {state.get('test_rows')}")
            except Exception as e:
                print(f"Error occurred while fetching test rows: {e}")

            if save_tc:
                rows = load_session_rows(session_id) or state.get("test_rows") 
                print(f"Test Rows to save: {rows}")
                if not rows:
                    state["messages"].append(AIMessage(content="⚠️ No test cases available to save. Generate first."))
                else:
                    saved_paths = []
                    if "csv" in [str(x).lower() for x in (save_tc or [])]:
                        path = save_execute_agent.save_testcase_excel_csv(transform="csv", data=rows)
                        state["messages"].append(AIMessage(content=f"✅ Saved CSV: {path}"))
                        saved_paths.append({"type": "csv", "path": path})
                    if any(x in [str(y).lower() for y in (save_tc or [])] for x in ["excel", "xlsx"]):
                        path = save_execute_agent.save_testcase_excel_csv(transform="excel", data=rows)
                        state["messages"].append(AIMessage(content=f"✅ Saved Excel: {path}"))
                        saved_paths.append({"type": "excel", "path": path})
                    try:
                        log_json_artifact(rows, "outputs/generated_testcases.json")
                        log_json_artifact(saved_paths, "outputs/saved_testcases_paths.json")

                        log_metrics({"test_rows_count": float(len(rows))})
                    except Exception:
                        pass

            # Final response (pre-moderation)
            response_text = state["messages"][-1].content
            print(f"response text : {response_text}")

            # Llama Guard 3 output moderation
            if plan_generation and "testcase" in plan_generation:
                try:
                    mod = self.moderator.enforce_or_refuse(response_text, "output", session_id)
                    if not mod.get("allowed", True):
                        response_text = "I can’t assist with that request."
                        state["messages"][-1] = AIMessage(content=response_text)
                        log_metrics({"moderation_forced_refusal": 1.0})
                    else:
                        log_metrics({"moderation_forced_refusal": 0.0})
                except Exception:
                    pass

            # Persist memory
            try: memory.save_context({"input": user_input}, {"output": response_text})
            except Exception: pass

            # MLflow timings
            try:
                node_latency = time.time() - t_node0
                log_text_artifact(response_text or "", "outputs/node_last_answer.txt")
                log_metrics({"node_total_latency_sec": float(node_latency)})
                for k, v in branch_latency.items():
                    log_metrics({f"{k}_latency_sec": float(v)})
                log_params({"messages_len": len(state.get("messages", []))})
            except Exception:
                pass

            return state

        except Exception as e:
            print(f"Error processing QEA Assistant: {e}")
            try:
                log_text_artifact(str(e), "errors/qea_assistant_error.txt")
                log_params({"node_last_error_type": e.__class__.__name__})
            except Exception:
                pass
            return {
                "messages": [{"type": "assistant", "content": f"Error processing request: {str(e)}"}],
                "session_id": session_id
            }
