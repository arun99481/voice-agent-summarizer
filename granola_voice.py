import json
import os
import operator
from typing import Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from plyer import notification
from dotenv import load_dotenv
load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")


# 1. Define the State
class State(TypedDict):
    user_input: str
    summary: str
    parsed_data: dict
    required_actions: Annotated[list, operator.add]
    response: Annotated[list, operator.add]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- NODES ---

def summarize_node(state: State):
    # Define the schema for structured output
    json_schema = {
        "title": "meeting_extraction",
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief overview of the meeting"},
            "tasks": {"type": "array", "items": {"type": "string"}, "description": "List of todos"},
            "followup": {"type": "string", "description": "Details of the next meeting or followup"}
        },
        "required": ["summary", "tasks", "followup"]
    }

    # Bind the schema to the LLM
    structured_llm = llm.with_structured_output(json_schema)
    
    prompt = "Analyze the meeting text and extract the summary, tasks, and followups."
    
    # Invoke and get a clean Python dictionary
    res = structured_llm.invoke([
        SystemMessage(content=prompt), 
        HumanMessage(content=state["user_input"])
    ])
    
    # Determine routing
    actions = []
    if res.get("tasks"): actions.append("todo")
    if res.get("followup"): actions.append("followup")
    
    return {
        "summary": res["summary"],
        "parsed_data": res, # Store the whole dict for the next nodes
        "required_actions": actions
    }

def todo_node(state: State):
    data = state["parsed_data"]
    tasks = data.get("tasks", [])
    
    if tasks:
        with open("todo_list.txt", "a") as f:
            f.write("\n--- New Tasks ---\n")
            for task in tasks:
                f.write(f"* {task}\n")
        return {"response": [f"Logged {len(tasks)} tasks."]}
    return {"response": []}

def followup_node(state: State):
    data = state["parsed_data"]
    followup = data.get("followup", "")
    
    if followup:
        os.system(f"osascript -e 'display notification \"{followup}\" with title \"Meeting Followup\"'")
        with open("reminders.txt", "a") as f:
            f.write(f"REMINDER: {followup}\n")
        return {"response": ["Created desktop reminder."]}
    return {"response": []}

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