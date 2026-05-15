from langchain_core.messages import (
    BaseMessage,
    HumanMessage
)
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph, MessagesState, START
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.types import Command
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langgraph_compare import *

# Create the experiment folder and SQLite checkpointer for run logging.
exp = create_experiment("network")
memory = exp.memory

# Load API keys from .env (OPENAI_API_KEY, TAVILY_API_KEY).
load_dotenv()

# Tavily web search — the researcher uses this to fetch real-time data.
tavily_tool = TavilySearchResults(max_results=1)

# Python REPL — the chart generator uses this to execute matplotlib code.
# Warning: executes code locally; unsafe outside a sandbox.
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# Shared system prompt injected into every agent.
# Agents signal completion by prefixing their response with "FINAL ANSWER".
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

llm = ChatOpenAI(model="gpt-4o")

# Routing helper: if the last message contains "FINAL ANSWER" the work is done,
# otherwise pass control to the next agent.
def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto


# Research agent — equipped with Tavily to search the web.
# Its node returns a Command that routes to chart_generator (or END).
research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    state_modifier=make_system_prompt(
        "You can only do research. You are working with a chart generator colleague."
    ),
)


def research_node(
    state: MessagesState,
) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_generator")
    # Re-wrap the last AI message as HumanMessage — some providers reject
    # an AI message in the last position of the input list.
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    # Command updates shared state with the agent's messages and sets the next node.
    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )


# Chart generator agent — equipped with a Python REPL to produce charts.
# Its node returns a Command that routes back to researcher (or END).
# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    state_modifier=make_system_prompt(
        "You can only generate charts. You are working with a researcher colleague."
    ),
)

def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")
    # Re-wrap for the same provider-compatibility reason as research_node.
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )


# Network (peer-to-peer) graph: no central supervisor.
# Agents hand off directly to each other via Command.goto until one signals FINAL ANSWER.
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

# Researcher always goes first; routing from there is driven by Command returns.
workflow.add_edge(START, "researcher")
graph = workflow.compile(checkpointer=memory)

user_input = {
    "messages": [
        HumanMessage(
            content="First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                "Once you make the chart, finish."
        )
    ]
}

print()
run_multiple_iterations(graph=graph, starting_thread_id=1, num_repetitions=3, user_input_template=user_input,
                        recursion_limit=150)
print()

# Both nodes must be declared so the exporter captures all events.
graph_config = GraphConfig(
    nodes=['researcher', 'chart_generator']
)

# ETL pipeline: SQLite → JSON → CSV event log.
prepare_data(exp, graph_config)

# Load the event log and print all process mining metrics.
print()
event_log = load_event_log(exp)
print_analysis(event_log)
print()

# Generate JSON reports and PNG visualizations (mermaid, prefix tree, DFG).
generate_artifacts(event_log, graph, exp)
