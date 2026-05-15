from dotenv import load_dotenv
from typing import Annotated

from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph_compare import *

# Create the experiment folder structure (db, json, csv, img, reports)
# and a SQLite checkpointer that LangGraph uses to persist run logs.
exp = create_experiment("main")
memory = exp.memory

# Load API keys from .env (OPENAI_API_KEY, LANGSMITH_API_KEY, etc.)
load_dotenv()

# State is the shared data structure passed between nodes.
# `add_messages` means new messages are appended to the list, not overwritten.
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# The LLM that powers the chatbot node.
llm = ChatOpenAI(model="gpt-4o-mini")

# The single node in this graph — receives the current state,
# calls the LLM, and returns the response as a new message.
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Register the node and wire it between START and END —
# this is a linear single-step graph with no branching.
graph_builder.add_node("chatbot_node", chatbot)
graph_builder.add_edge(START, "chatbot_node")
graph_builder.add_edge("chatbot_node", END)

# Compile the graph with the SQLite checkpointer so every run is logged.
graph = graph_builder.compile(checkpointer=memory)

# Run the graph 5 times (thread IDs 1–5) with the same input.
# Each iteration is an independent conversation thread stored in SQLite.
print()
run_multiple_iterations(graph, 1, 5, {"messages": [("user", "Tell me a joke")]})
print()

# Tell the exporter which nodes belong to this graph,
# then run the full ETL pipeline: SQLite → JSON → CSV.
graph_config = GraphConfig(nodes=["chatbot_node"])
prepare_data(exp, graph_config)

# Load the CSV event log and print all process mining metrics.
print()
event_log = load_event_log(exp)
print_analysis(event_log)
print()

# Generate JSON reports and PNG visualizations (mermaid, prefix tree, DFG).
generate_artifacts(event_log, graph, exp)