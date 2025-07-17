from src.mutli_agent_langgraph.state.state import State
from src.mutli_agent_langgraph.memory.langchain_conversation import LangchainConversation
from langchain_core.messages import BaseMessage,SystemMessage,AIMessage
from src.mutli_agent_langgraph.agents import testcase_agent_,testscript_agent,userstory_agent
class QEAAssistantChatbot:

    def __init__(self,model):
        self.llm = model

    def process(self,state: State)-> dict :
        """
        Process the input state and generate a qea chatbot response
        """

        session_id = state["session_id"]
        
        user_input = state["messages"][-1].content

        conversation_memory = LangchainConversation(session_id=session_id)
        memory = conversation_memory.get_conversation_memory()
        chat_history = memory.load_memory_variables({})['history']
        print(f"Chat History: {chat_history}")
        try:
            history = "\n".join(f"{message.type} : {message.content}" for message in chat_history)
            prompt = f"{history}\nuser: {user_input}\nassistant:"


            plan = state["plan"].generation
            retriver = state['retrived_content']

            print(f"Retriver which i got from node: {retriver}")

            plan_irrelevant = state["plan"].irrelevant
            plan_chitchat = state["plan"].chitchat

            
            if "testcase" in plan:
                testcase_agent_.testcase(retriver=retriver,user_message=prompt,model=self.llm,state=state)
            elif "testscript" in plan:
                testscript_agent.testscript(retriver=retriver,user_message=prompt,model=self.llm,state=state)
            elif "userstory" in plan:
                userstory_agent.userstory(retriver=retriver,user_message=prompt,model=self.llm,state=state)
            elif plan_irrelevant :
                state["messages"].append(AIMessage(content=plan_irrelevant))
            elif plan_chitchat:
                state["messages"].append(AIMessage(content=plan_chitchat))
            else:
                state["messages"] = None


            response_text = state["messages"][-1].content
            print(f"response text : {response_text}")

            # response_text = ""
            
            # for chunks in self.llm.stream(prompt):
            #     if chunks.content:
            #         response_text  += chunks.content

            memory.save_context({"input": user_input}, {"output": response_text})

            return {
            "messages":state["messages"],
            "session_id": session_id
            }
        except Exception as e:
            print(f"Error processing QEA Assistant: {e}")
            return {
                "messages": [{"type": "assistant", "content": f"Error processing request: {str(e)}"}],
                "session_id": session_id
            }
        
    
    def create_chatbot(self,tools):
        """
        Returns a chatbot node fucntion
        """

        llm_with_tools =self.llm.bind_tools(tools)

        def chatbot_node(state: State)->dict:
            session_id = state["session_id"]

            # ---- 1. Load past memory ----
            conv_mem  = LangchainConversation(session_id)
            past      = conv_mem.get_conversation_memory().load_memory_variables({})["history"]

            # ---- 2. Merge with incoming turn ----
            # Keep everything we already have in this turn!
            working_ctx: list[BaseMessage] = past + state["messages"]


            # ---- 3. Let the LLM (with tools) respond ----
            ai_msg: BaseMessage = llm_with_tools.invoke(working_ctx)
            working_ctx.append(ai_msg)                # ⬅️ very important

            # ---- 4. Persist ONLY if it’s the final answer ----
            # If the assistant just asked for a tool call, ai_msg.content == ""

           # 4. persist only final answer
            if not getattr(ai_msg, "tool_calls", None):
                # find the most recent HumanMessage
                last_human = next(
                    (m for m in reversed(working_ctx) if m.type == "human"), None
                )
                if last_human:                              # guard in case something weird
                    conv_mem.get_conversation_memory().save_context(
                        {"input": last_human.content},      # human question
                        {"output": ai_msg.content}          # assistant answer
                    )

            # 5. return full context
            return {
                "messages": working_ctx,
                "session_id": session_id
            }
        return chatbot_node