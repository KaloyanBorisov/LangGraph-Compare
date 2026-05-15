# REQUIRES BUMP FROM 3.9 -> 3.11?

from dotenv import load_dotenv
from typing import Annotated
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal
import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from typing import Union
import functools as _functools

from langgraph_compare import *

# Create experiment folder and SQLite checkpointer for run logging.
exp = create_experiment("supervision")
memory = exp.memory

# Load API keys from .env (OPENAI_API_KEY, TAVILY_API_KEY).
load_dotenv()

# Tavily for real-time web search; PythonREPLTool for executing code locally.
# Warning: PythonREPLTool runs arbitrary code — unsafe outside a sandbox.
tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()


# Each worker agent returns its result wrapped as a HumanMessage so the
# supervisor sees it as a plain message, not an AI response.
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }


# The supervisor chooses which worker acts next, or signals FINISH when done.
# options lists all valid routing targets including the terminal state.
members = ["Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options = ["FINISH"] + members

# Python 3.10-compatible dynamic Literal: reduce options list into a Union of Literals.
# Equivalent to Literal["FINISH", "Researcher", "Coder"] but built at runtime.
_NextLiteral = _functools.reduce(lambda a, b: Union[a, Literal[b]], options[1:], Literal[options[0]])

# Structured output schema — the supervisor LLM must return exactly this shape,
# forcing it to pick one of the valid options as the next worker.
class routeResponse(BaseModel):
    next: _NextLiteral


# Prompt feeds the supervisor the full conversation history plus a final
# instruction to pick the next worker or FINISH.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# gpt-4o used here (not mini) because the supervisor requires reliable structured output.
llm = ChatOpenAI(model="gpt-4o")


# Supervisor node: runs the prompt → LLM pipeline and returns the routing decision.
# with_structured_output forces the LLM to return a valid routeResponse JSON.
def supervisor_agent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    return supervisor_chain.invoke(state)


# Shared state: accumulated message history plus the supervisor's current routing decision.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# Researcher agent — searches the web with Tavily.
research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# Coder agent — writes and executes Python code via the REPL.
code_agent = create_react_agent(llm, tools=[python_repl_tool])
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)

# Every worker always reports back to the supervisor after finishing its task.
for member in members:
    workflow.add_edge(member, "supervisor")

# The supervisor's "next" field drives the conditional routing:
# "Researcher" → Researcher node, "Coder" → Coder node, "FINISH" → END.
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Supervisor is the entry point — it decides who acts first.
workflow.add_edge(START, "supervisor")

graph = workflow.compile(checkpointer=memory)

# Run 3 iterations of a coding task.
user_input = {
    "messages": [
        HumanMessage(
            content="Code hello world and print it to the terminal"
        )
    ]
}

run_multiple_iterations(graph=graph, starting_thread_id=1, num_repetitions=3, user_input_template=user_input,
                        recursion_limit=100)

# Run 3 more iterations of a research task (thread IDs 4–6).
user_input = {
    "messages": [
        HumanMessage(
            content="Write a brief research report on pikas."
        )
    ]
}

print()
run_multiple_iterations(graph=graph, starting_thread_id=4, num_repetitions=3, user_input_template=user_input,
                        recursion_limit=100)
print()

# supervisor is the graph-level supervisor; Researcher and Coder are worker nodes.
supervisor = SupervisorConfig(
    name="supervisor",
    supervisor_type="graph"
)

graph_config = GraphConfig(
    supervisors=[supervisor],
    nodes=["Researcher", "Coder"]
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
