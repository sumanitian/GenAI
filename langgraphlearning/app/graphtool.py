from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
load_dotenv()

# @tool()
# def human_assistance_tool(query: str):
#     """Request assistance from a human."""
#     human_response = interrupt({ "query": query }) # Graph will exit out after saving data in DB
#     return human_response["data"] # resume with the data

@tool()
def human_assistance_tool(query: str):
    """
    Use this tool ONLY for performing real-world actions like:
    - updating user data
    - modifying account details
    - executing system changes

    Do NOT use for general conversation.
    """
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [human_assistance_tool]

llm = init_chat_model(model_provider="groq", model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools=tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]

# def chatbot(state: State):
#     message = llm_with_tools.invoke(state["messages"])
#     assert len(message.tool_calls) <= 1
#     return {"messages": [message]}

def chatbot(state: State):
    SYSTEM_PROMPT = """
    You are an AI assistant.

    Rules:
    1. DO NOT call any tool for normal conversation or storing information.
    2. ONLY call the human_assistance_tool when the user requests a real-world action like:
        - changing account number
        - updating personal details
        - deleting account
        - modifying system data
    3. If the user is just giving information (like name, account number), DO NOT call the tool.
    4. If the request involves system/database changes, you MUST call the tool.
    5. After receiving tool response, provide final answer and DO NOT call tool again.

    Examples:
    - "My name is Suman" → Response: Got it, Suman!
    - "What is my name?" → → Response: Your name is Suman.
    - "Change my account number" → CALL TOOL
    """

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]

    message = llm_with_tools.invoke(messages)

    return {"messages": [message]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot") # We missed this

graph_builder.add_edge("chatbot", END)

# Without any memory
graph = graph_builder.compile()

# Creates a new graph with given checkpointer
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
    