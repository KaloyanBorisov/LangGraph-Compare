import functools
import operator
import os

from typing import Annotated, List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from pathlib import Path
from tempfile import TemporaryDirectory
from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict, NotRequired
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage, trim_messages, BaseMessage
from langgraph.prebuilt import create_react_agent

from langgraph_compare import *

# Delete previous run data so create_experiment doesn't raise FileExistsError.
import shutil
_exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "hierarchical")
if os.path.exists(_exp_dir):
    shutil.rmtree(_exp_dir)

# Create experiment folder and SQLite checkpointer for run logging.
exp = create_experiment("hierarchical")
memory = exp.memory

# Load API keys from .env (OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY).
load_dotenv()

# ─── LANGSMITH TRACING ────────────────────────────────────────────────────────
# Requires LANGSMITH_API_KEY and optionally LANGCHAIN_PROJECT in your .env.
# LANGCHAIN_TRACING_V2=true enables automatic tracing of all LangChain/LangGraph calls.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "hierarchical-teams")

# ─── TOOLS ────────────────────────────────────────────────────────────────────

# Tavily for real-time web search (Research Team).
tavily_tool = TavilySearchResults(max_results=5)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


# Shared temp directory — all document writing tools read/write files here.
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)


@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to save the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    return f"Document edited and saved to {file_name}"


# Python REPL for chart generation (Paper Writing Team).
# Warning: executes code locally — unsafe outside a sandbox.
repl = PythonREPL()


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"


# ─── SHARED HELPERS ───────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini")

# Token trimmer keeps conversation history under 100k tokens so the LLM
# doesn't exceed its context window on long multi-agent runs.
trimmer = trim_messages(
    max_tokens=100000,
    strategy="last",
    token_counter=llm,
    include_system=True,
)


# Each worker agent wraps its final message as HumanMessage so the
# team supervisor receives it as a plain user turn, not an AI response.
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }


def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """Build a team supervisor chain using structured output."""
    from pydantic import BaseModel, Field
    from typing import Literal

    options = ["FINISH"] + members

    class RouteSchema(BaseModel):
        next: Literal[tuple(options)] = Field(description="The next role to act, or FINISH.")  # type: ignore[valid-type]

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
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | trimmer
        | llm.with_structured_output(RouteSchema)
        | (lambda x: {"next": x.next})
    )


# ─── RESEARCH TEAM ────────────────────────────────────────────────────────────
# A self-contained subgraph with its own supervisor (rg_supervisor).
# Workers: Search (Tavily) and WebScraper (HTML scraping).

llm = ChatOpenAI(model="gpt-4o")

# Shared state for the Research Team subgraph.
class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: NotRequired[str]  # supervisor routing decision


search_agent = create_react_agent(llm, tools=[tavily_tool])
search_node = functools.partial(agent_node, agent=search_agent, name="Search")

research_agent = create_react_agent(llm, tools=[scrape_webpages])
research_node = functools.partial(agent_node, agent=research_agent, name="WebScraper")

supervisor_agent = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  Search, WebScraper. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Search", "WebScraper"],
)

research_graph = StateGraph(ResearchTeamState)
research_graph.add_node("Search", search_node)
research_graph.add_node("WebScraper", research_node)
research_graph.add_node("rg_supervisor", supervisor_agent)

# Workers always report back to their team supervisor.
research_graph.add_edge("Search", "rg_supervisor")
research_graph.add_edge("WebScraper", "rg_supervisor")
research_graph.add_conditional_edges(
    "rg_supervisor",
    lambda x: x.get("next", "FINISH"),
    {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
)
research_graph.add_edge(START, "rg_supervisor")
_ckpt = memory if __name__ == "__main__" else None
chain = research_graph.compile(checkpointer=_ckpt)


# Adapter: the top-level graph passes a plain string; this wraps it into
# the ResearchTeamState message format expected by the subgraph.
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


research_chain = enter_chain | chain


# ─── PAPER WRITING TEAM ───────────────────────────────────────────────────────
# A self-contained subgraph with its own supervisor (ag_supervisor).
# Workers: DocWriter, NoteTaker, ChartGenerator — all share WORKING_DIRECTORY.

# Shared state for the Paper Writing Team subgraph.
class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: NotRequired[str]  # supervisor routing decision
    current_files: str  # snapshot of files written so far


# Injected before each worker call so agents are aware of what's already on disk.
def prelude(state):
    written_files = []
    if not WORKING_DIRECTORY.exists():
        WORKING_DIRECTORY.mkdir()
    try:
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
        ]
    except Exception:
        pass
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }


llm = ChatOpenAI(model="gpt-4o")

# Each writing agent is wrapped with prelude so it sees the current file listing.
doc_writer_agent = create_react_agent(
    llm, tools=[write_document, edit_document, read_document]
)
context_aware_doc_writer_agent = prelude | doc_writer_agent
doc_writing_node = functools.partial(
    agent_node, agent=context_aware_doc_writer_agent, name="DocWriter"
)

note_taking_agent = create_react_agent(llm, tools=[create_outline, read_document])
context_aware_note_taking_agent = prelude | note_taking_agent
note_taking_node = functools.partial(
    agent_node, agent=context_aware_note_taking_agent, name="NoteTaker"
)

chart_generating_agent = create_react_agent(llm, tools=[read_document, python_repl])
context_aware_chart_generating_agent = prelude | chart_generating_agent
chart_generating_node = functools.partial(
    agent_node, agent=context_aware_note_taking_agent, name="ChartGenerator"
)

doc_writing_supervisor = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["DocWriter", "NoteTaker", "ChartGenerator"],
)

authoring_graph = StateGraph(DocWritingState)
authoring_graph.add_node("DocWriter", doc_writing_node)
authoring_graph.add_node("NoteTaker", note_taking_node)
authoring_graph.add_node("ChartGenerator", chart_generating_node)
authoring_graph.add_node("ag_supervisor", doc_writing_supervisor)

authoring_graph.add_edge("DocWriter", "ag_supervisor")
authoring_graph.add_edge("NoteTaker", "ag_supervisor")
authoring_graph.add_edge("ChartGenerator", "ag_supervisor")
authoring_graph.add_conditional_edges(
    "ag_supervisor",
    lambda x: x.get("next", "FINISH"),
    {
        "DocWriter": "DocWriter",
        "NoteTaker": "NoteTaker",
        "ChartGenerator": "ChartGenerator",
        "FINISH": END,
    },
)
authoring_graph.add_edge(START, "ag_supervisor")
chain = authoring_graph.compile(checkpointer=_ckpt)


# Adapter: wraps the top-level string message and injects team_members list
# into DocWritingState before entering the subgraph.
def enter_chain(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results


authoring_chain = (
    functools.partial(enter_chain, members=authoring_graph.nodes)
    | authoring_graph.compile(checkpointer=_ckpt)
)


# ─── TOP-LEVEL GRAPH ──────────────────────────────────────────────────────────
# A meta-supervisor (test_supervisor) routes between the two team subgraphs.
# This is the hierarchical layer: supervisor-of-supervisors pattern.

from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

supervisor_node = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following teams: {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["ResearchTeam", "PaperWritingTeam"],
)

# Top-level state: shared message history and the meta-supervisor's routing decision.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: NotRequired[str]


def _msg_content(msg) -> str:
    return msg.content if hasattr(msg, "content") else msg["content"]


def get_last_message(state: State) -> str:
    return _msg_content(state["messages"][-1])


def join_graph(response: dict):
    last = response["messages"][-1]
    if not hasattr(last, "content"):
        last = HumanMessage(content=last["content"])
    return {"messages": [last]}


super_graph = StateGraph(State)
# Each team node is a pipeline: extract message → run subgraph → collapse output.
super_graph.add_node("ResearchTeam", get_last_message | research_chain | join_graph)
super_graph.add_node(
    "PaperWritingTeam", get_last_message | authoring_chain | join_graph
)
super_graph.add_node("test_supervisor", supervisor_node)

# Teams always report back to the meta-supervisor after finishing.
super_graph.add_edge("ResearchTeam", "test_supervisor")
super_graph.add_edge("PaperWritingTeam", "test_supervisor")
super_graph.add_conditional_edges(
    "test_supervisor",
    lambda x: x.get("next", "FINISH"),
    {
        "PaperWritingTeam": "PaperWritingTeam",
        "ResearchTeam": "ResearchTeam",
        "FINISH": END,
    },
)
super_graph.add_edge(START, "test_supervisor")
super_graph = super_graph.compile(checkpointer=_ckpt)

if __name__ == "__main__":
    user_input = {
        "messages": [
            HumanMessage(
                content="Write a brief research report on the North American sturgeon. Include a chart."
            )
        ]
    }

    print()
    run_multiple_iterations(graph=super_graph, starting_thread_id=1, num_repetitions=1, user_input_template=user_input,
                            recursion_limit=150)
    print()

    # ─── PROCESS MINING CONFIG ────────────────────────────────────────────────────
    # Three supervisor levels: one graph-level meta-supervisor and two subgraph supervisors.

    test_supervisor = SupervisorConfig(
        name="test_supervisor",
        supervisor_type="graph"
    )

    rg_supervisor = SupervisorConfig(
        name="rg_supervisor",
        supervisor_type="subgraph"
    )

    ag_supervisor = SupervisorConfig(
        name="ag_supervisor",
        supervisor_type="subgraph"
    )

    # ResearchTeam subgraph: Search + WebScraper workers under rg_supervisor.
    research_team = SubgraphConfig(
        name="ResearchTeam",
        nodes=["Search", "WebScraper"],
        supervisor=rg_supervisor
    )

    # PaperWritingTeam subgraph: DocWriter + NoteTaker + ChartGenerator under ag_supervisor.
    authoring_team = SubgraphConfig(
        name="PaperWritingTeam",
        nodes=["DocWriter", "NoteTaker", "ChartGenerator"],
        supervisor=ag_supervisor
    )

    graph_config = GraphConfig(
        supervisors=[test_supervisor],
        subgraphs=[research_team, authoring_team]
    )

    # ETL pipeline: SQLite → JSON → CSV event log.
    prepare_data(exp, graph_config)

    # Load the event log and print all process mining metrics.
    print()
    event_log = load_event_log(exp)
    print_analysis(event_log)
    print()

    # Generate JSON reports and PNG visualizations (mermaid, prefix tree, DFG).
    generate_artifacts(event_log, super_graph, exp)
