from dotenv import load_dotenv
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_compare import *

# Create experiment folder structure and SQLite checkpointer for run logging.
exp = create_experiment("tavily")
memory = exp.memory

# Load API keys from .env (OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY).
load_dotenv()

# State holds the conversation message history.
# add_messages appends new messages instead of overwriting.
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# Tavily is a real-time web search tool — the LLM calls it when it needs
# up-to-date information it doesn't have in its training data.
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Bind the tool to the LLM so it knows it can call it and how to call it.
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


# The chatbot node — invokes the LLM which may respond directly
# or emit a tool call if it needs to search the web.
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot_node", chatbot)

# ToolNode executes whichever tool the LLM requested and returns the result.
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# tools_condition routes the flow:
#   - if the LLM emitted a tool call → go to "tools"
#   - if the LLM gave a final answer → go to END
graph_builder.add_conditional_edges("chatbot_node", tools_condition)

# After the tool runs, always return to the chatbot so the LLM can
# incorporate the search result and decide what to do next.
graph_builder.add_edge("tools", "chatbot_node")
graph_builder.set_entry_point("chatbot_node")

# Compile with checkpointer so every step is persisted to SQLite.
graph = graph_builder.compile(checkpointer=memory)

# Run 3 iterations with the same question — each is an independent thread.
# The LLM will search Tavily for current info about PJATK before answering.
user_input = {"messages": [("user", "Tell me about PJATK in Warsaw")]}

print()
run_multiple_iterations(graph, 1, 3, user_input)
print()

# Both nodes must be declared so the exporter captures all events.
graph_config = GraphConfig(nodes=["chatbot_node", "tools"])

# ETL pipeline: SQLite → JSON → CSV event log.
prepare_data(exp, graph_config)

# Load the event log and print all process mining metrics.
print()
event_log = load_event_log(exp)
print_analysis(event_log)
print()

# Generate JSON reports and PNG visualizations (mermaid, prefix tree, DFG).
generate_artifacts(event_log, graph, exp)