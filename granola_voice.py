import operator
from typing import Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("GEMINI_API_KEY")

# 1. Define the State
class State(TypedDict):
    user_input: str
    summary: str
    # This stores the decisions: e.g., ["TODO", "FOLLOWUP"]
    required_actions: Annotated[list, operator.add]
    response: Annotated[list, operator.add]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- NODES ---

def summarize_node(state: State):
    prompt = """Summarize the following text. 
    Then, detect if any 'Tasks/Todos' or 'Followup Meetings' are mentioned.
    Return your response in this EXACT format:
    SUMMARY: (your summary)
    ACTIONS: [TODO] [FOLLOWUP] (only include the brackets that apply)"""
    
    res = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["user_input"])])
    content = res.content
    
    # Simple parsing logic
    actions = []
    if "[TODO]" in content: actions.append("todo")
    if "[FOLLOWUP]" in content: actions.append("followup")
    
    return {
        "summary": content.split("ACTIONS:")[0],
        "required_actions": actions
    }

def todo_node(state: State):
    print("--- Executing TODO Node ---")
    return {"response": ["Todo item has been logged."]}

def followup_node(state: State):
    print("--- Executing FOLLOWUP Node ---")
    return {"response": ["Followup meeting has been scheduled."]}

# --- ROUTER ---

def router(state: State) -> List[str]:
    # This is the magic: return a list of all nodes that should run
    # If the list is empty, it goes nowhere (ends).
    actions = state["required_actions"]
    if not actions:
        return [END]
    return actions

# --- GRAPH ---

workflow = StateGraph(State)

workflow.add_node("summarize", summarize_node)
workflow.add_node("todo", todo_node)
workflow.add_node("followup", followup_node)

workflow.add_edge(START, "summarize")

# Conditional edges can point to multiple destinations simultaneously
workflow.add_conditional_edges(
    "summarize",
    router,
    {
        "todo": "todo",
        "followup": "followup",
        END: END
    }
)

workflow.add_edge("todo", END)
workflow.add_edge("followup", END)

app = workflow.compile()