from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal

class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool

# routing node
def detect_query(state: State):
    user_message = state.get("user_message")

    #openai call

    state["is_coding_question"] = False
    return state

# conditional edge
def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_simple_question"

def solve_coding_question(state: State):
    user_message = state.get("user_message")

    #openai call (coding question)
    state["ai_message"] = "Here is you coding question ans"

    return state

def solve_simple_question(state: State):
    user_message = state.get("user_message")

    #openai call (coding question)
    state["ai_message"] = "Please ask some coding question"

    return state

graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)
graph_builder.add_node("route_edge", route_edge)


# start building edge

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)


graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()

#use the Graph

def call_graph():
    state = {
        "user_message": "Hello, how are you?",
        "ai_message": "",
        "is_coding_question": False
    }
    result = graph.invoke(state)
    print("Final Result", result)

call_graph()