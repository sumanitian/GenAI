import os
import json
import re
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Schema
class DetectCallResponse(BaseModel):
    is_question_ai: bool

class CodingAIResponse(BaseModel):
    answer: str

# ✅ Groq client
client = wrap_openai(OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
))

class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool


# ------------------ detect_query ------------------
def detect_query(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is related
    to coding question or not.

    Return ONLY JSON:
    {"is_question_ai": true or false}
    """

    result = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ],
        temperature=0
    )

    content = result.choices[0].message.content
    print("RAW detect:", content)

    # clean JSON if wrapped in ```
    content = re.sub(r"```json|```", "", content).strip()

    parsed = json.loads(content)

    state["is_coding_question"] = parsed["is_question_ai"]
    return state


# ------------------ routing ------------------
def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_simple_question"


# ------------------ coding solver ------------------
def solve_coding_question(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to resolve the user query based on coding 
    problem he is facing
    """

    result = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ]
    )

    state["ai_message"] = result.choices[0].message.content
    return state


# ------------------ simple chat ------------------
def solve_simple_question(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to chat with user
    """

    result = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ]
    )

    state["ai_message"] = result.choices[0].message.content
    return state


# ------------------ Graph ------------------
graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)
graph_builder.add_node("route_edge", route_edge)  # kept same as your code

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)

graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()


# ------------------ Run ------------------
def call_graph():
    state = {
        "user_message": "Explain Pydantic in python?",
        "ai_message": "",
        "is_coding_question": True
    }

    result = graph.invoke(state)
    print("Final Result", result)


call_graph()