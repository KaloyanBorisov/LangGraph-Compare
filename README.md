# LangGraph Compare

A Python package for **benchmarking, analyzing, and comparing LangGraph multi-agent architectures** using process mining techniques.

## Documentation

Full API reference and guides: https://kaloyanborisov.github.io/LangGraph-Compare/

## Comparison Reports

Interactive HTML reports comparing all architecture experiments (hosted on GitHub Pages):

| Report | Architectures |
|---|---|
| [All architectures](https://kaloyanborisov.github.io/LangGraph-Compare/comparison_reports/main_vs_tavily_vs_supervision_vs_hierarchical_vs_network_vs_mcp_math_100_vs_mcp_docs_100.html) | main, tavily, supervision, hierarchical, network, mcp_math, mcp_docs |
| [Without MCP](https://kaloyanborisov.github.io/LangGraph-Compare/comparison_reports/main_vs_tavily_vs_supervision_vs_hierarchical_vs_network.html) | main, tavily, supervision, hierarchical, network |
| [main vs tavily](https://kaloyanborisov.github.io/LangGraph-Compare/comparison_reports/main_vs_tavily.html) | main, tavily |
| [Reflexion variants](https://kaloyanborisov.github.io/LangGraph-Compare/comparison_reports/programming_100_vs_climate_100_vs_math_100.html) | programming, climate, math |

## LangGraph Studio

Interactively debug the hierarchical multi-agent architecture in [LangGraph Studio](https://smith.langchain.com/studio/?baseUrl=http://localhost:3000).

**Requirements:** Docker container running with `langgraph dev` (see [Development Setup](#development-setup-poetry)).

**Steps:**

1. Start the dev container:
   ```bash
   docker compose up -d dev
   docker exec -it langgraph-compare-dev bash
   ```
2. Inside the container, start the LangGraph API server:
   ```bash
   langgraph dev --host 0.0.0.0 --no-reload
   ```
3. Open Studio in your browser (server must be running):
   [https://smith.langchain.com/studio/?baseUrl=http://localhost:3000](https://smith.langchain.com/studio/?baseUrl=http://localhost:3000)

**Example prompts:**
- `Research the latest developments in quantum computing and summarize the key findings.`
- `Research the current state of large language models and write a short report about the most important recent advances.`
- `Research the market share of major cloud providers (AWS, Azure, GCP) and create a chart comparing them.`

---

## Overview

When LangGraph agents execute, their run logs are stored in an SQLite database in a binary (msgpack-encoded) format. `langgraph_compare` provides a complete pipeline to:

1. **Extract** — decode and export run logs from SQLite to JSON
2. **Transform** — convert JSON event data into a structured CSV event log
3. **Analyze** — apply process mining metrics (via [pm4py](https://pm4py.fit.fraunhofer.de/)) to the event log
4. **Visualize** — generate process flow diagrams and performance charts
5. **Compare** — produce an interactive HTML report comparing multiple architectures side by side

This makes it straightforward to answer questions like: _Which architecture executes faster? Which nodes are bottlenecks? How much rework does each agent do?_

---

## Architecture

```
LangGraph run (SQLite checkpoint DB)
        │
        ▼
export_sqlite_to_jsons()   ← decode msgpack, write per-thread JSON files
        │
        ▼
export_jsons_to_csv()      ← parse JSON events, emit standard event log CSV
        │
        ▼
load_event_log()           ← load CSV into pm4py-compatible DataFrame
        │
        ├─► print_analysis() / get_*()    ← process mining metrics
        ├─► generate_artifacts()          ← JSON reports + PNG visualizations
        └─► compare()                     ← cross-architecture HTML report
```

### Experiment folder structure

`create_experiment("name")` sets up this layout under `experiments/`:

```
experiments/
└── name/
    ├── db/       ← SQLite checkpoint database (langgraph writes here)
    ├── json/     ← one JSON file per conversation thread
    ├── csv/      ← csv_output.csv (the event log)
    ├── img/      ← mermaid.png, prefix_tree.png, dfg_performance.png
    └── reports/  ← metrics_report.json, sequences_report.json
```

---

## Installation

Python 3.9 or higher is required.

### System dependency: Graphviz

**Windows** — download from [graphviz.org](https://graphviz.org/download/)

**macOS**
```bash
brew install graphviz
```

**Debian / Ubuntu**
```bash
sudo apt-get install graphviz
```

**Fedora / RHEL / Rocky / CentOS**
```bash
sudo dnf install graphviz
```

### Package

```bash
pip install langgraph_compare
```

Or with conda:

```bash
conda create -n langgraph_compare python=3.9
conda activate langgraph_compare
pip install langgraph_compare
```

### Development setup (Poetry)

Requires Python 3.10+ (Sphinx minimum):

```bash
poetry install --with dev,test,docs
poetry run pytest
```

---

## Quick Start

### 1. Single architecture — basic chatbot

```python
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph_compare import *

# Create experiment folder structure and SQLite checkpointer
exp = create_experiment("my_chatbot")
memory = exp.memory

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot_node", chatbot)
graph_builder.add_edge(START, "chatbot_node")
graph_builder.add_edge("chatbot_node", END)
graph = graph_builder.compile(checkpointer=memory)

# Run 5 iterations (thread IDs 1–5) with the same input
run_multiple_iterations(graph, 1, 5, {"messages": [("user", "Tell me a joke")]})

# Extract, transform, and load the event log
graph_config = GraphConfig(nodes=["chatbot_node"])
prepare_data(exp, graph_config)

event_log = load_event_log(exp)
print_analysis(event_log)

# Generate JSON reports and PNG visualizations
generate_artifacts(event_log, graph, exp)
```

### 2. Compare multiple architectures

After running experiments for each architecture separately:

```python
from langgraph_compare import compare

compare(["supervise", "linear", "random"])
# → generates comparison_reports/supervise_vs_linear_vs_random.html
```

---

## Key Concepts

### `GraphConfig`

Tells the CSV exporter which nodes belong to your graph. For hierarchical or supervised architectures, use `SubgraphConfig` / `SupervisorConfig`:

```python
# Simple graph
GraphConfig(nodes=["node_a", "node_b"])

# Supervisor with subgraphs
GraphConfig(
    nodes=["supervisor"],
    supervisor=SupervisorConfig(name="supervisor"),
    subgraphs=[SubgraphConfig(name="research_team", nodes=["searcher", "writer"])]
)
```

### `run_multiple_iterations(graph, start_id, end_id, input)`

Runs the compiled graph for each thread ID from `start_id` to `end_id` inclusive, using the same input each time. Thread IDs map to LangGraph's `{"configurable": {"thread_id": ...}}`.

### `prepare_data(exp, graph_config)`

Convenience wrapper that calls `export_sqlite_to_jsons()` → `export_jsons_to_csv()` in sequence.

### `generate_artifacts(event_log, graph, exp)`

Convenience wrapper that calls `generate_reports()` → `generate_visualizations()` in sequence.

---

## Analysis Metrics

`print_analysis(event_log)` (or individual `get_*` / `print_*` functions) covers:

| Metric | Function |
|---|---|
| Start / end activities | `get_starts`, `get_ends` |
| Activity execution counts | `get_act_counts` |
| Per-case activity sequences | `get_sequences` |
| Sequence probabilities | `get_sequence_probs` |
| Minimum self-distances | `get_min_self_dists` |
| Self-distance witnesses | `get_self_dist_witnesses` |
| Per-case activity rework | `get_act_reworks` |
| Global activity rework | `get_global_act_reworks` |
| Mean activity service time | `get_mean_act_times` |
| Per-case duration | `get_durations` |
| Average case duration | `get_avg_duration` |

Per-case variants (prefix `get_case_*`) are also available for drilling into a single thread.

---

## Visualizations

`generate_visualizations(event_log, graph, exp)` produces three PNG files:

| File | Description |
|---|---|
| `mermaid.png` | LangGraph's built-in Mermaid graph of the compiled graph structure |
| `prefix_tree.png` | Prefix tree showing all observed execution paths |
| `dfg_performance.png` | Directly-Follows Graph annotated with mean service times |

Sample output:

![Sample DFG](docs/img/sample_dfg_performance.png)
![Sample Mermaid](docs/img/sample_mermaid.png)
![Sample Prefix Tree](docs/img/sample_tree.png)

---

## Supported Architecture Patterns

### Single-Agent / Linear Chain (`examples/architectures/main.py`)

```
START → chatbot_node → END
```

Single node, no tools, no routing. The LLM answers directly from its training data.

**Use case:** Simple Q&A where real-time data or actions are not needed.  
**Cost:** Lowest — one LLM call per run.

---

### ReAct — Reason + Act (`examples/architectures/main_tavily.py`)

```
START → chatbot_node ⇄ tools → END
```

ReAct loop — the LLM decides each turn whether to call Tavily web search or return a final answer. Can loop multiple times before finishing.

**Use case:** Questions requiring real-time or up-to-date information.  
**Cost:** Low to medium — scales with the number of search iterations.

---

### Supervisor / Hub-and-Spoke (`examples/architectures/main_supervision.py`)

```
START → supervisor → Researcher → supervisor
                  ↘ Coder     → supervisor
                  ↘ FINISH    → END
```

A central supervisor LLM routes between specialist workers. Workers always report back to the supervisor after each step. Routing is enforced via structured output (`routeResponse`).

**Use case:** Tasks that require both research (Tavily) and code execution (Python REPL).  
**Cost:** Medium to high — every worker call is preceded and followed by a supervisor call.

---

### Agent Network / Peer-to-Peer (`examples/architectures/main_network.py`)

```
START → researcher ⇄ chart_generator → END
```

Peer-to-peer — no central supervisor. Each agent returns a `Command` that directly sets the next node. Agents signal completion by prefixing their response with `FINAL ANSWER`.

**Use case:** Two complementary agents collaborating in sequence (research → visualize).  
**Cost:** Medium — no supervisor overhead, but agents may loop back and forth before finishing.

---

### Hierarchical Agent Teams (`examples/architectures/main_hierarchical_teams.py`)

```
START → test_supervisor → ResearchTeam    (subgraph: rg_supervisor → Search / WebScraper)
                       ↘ PaperWritingTeam (subgraph: ag_supervisor → DocWriter / NoteTaker / ChartGenerator)
                       ↘ FINISH → END
```

Three levels of supervision: a meta-supervisor routes between two full sub-teams, each of which has its own internal supervisor and worker agents. Sub-teams share a working directory for file-based handoffs.

**Use case:** Long-horizon complex tasks — research a topic, then write and chart a full paper.  
**Cost:** Highest — many LLM calls across 8 nodes and 3 supervisor layers.

---

### Summary

| Architecture | Nodes | Supervisor levels | Tools | Complexity | Cost |
|---|---|---|---|---|---|
| Basic Chatbot | 1 | 0 | None | Lowest | $ |
| Tavily | 2 | 0 | Web search | Low | $$ |
| MCP — Math (stdio) | 2 | 0 | Local arithmetic server | Low | $$ |
| MCP — Docs (HTTP) | 2 | 0 | Remote LangChain docs server | Low | $$ |
| Supervision | 3 | 1 | Search + Code | Medium | $$$ |
| Network | 2 | 0 (peer) | Search + Code | Medium | $$$ |
| Hierarchical Teams | 8 | 3 (nested) | Search + Scrape + Docs + Code | Highest | $$$$ |

The key design choice is how much coordination overhead you are willing to pay versus how complex the task is. Simpler architectures are cheaper and easier to debug; hierarchical architectures can tackle much harder tasks but multiply LLM calls at every level.

---

### MCP — Model Context Protocol (`examples/mcp/`)

```
START → agent ⇄ tools → END
```

A standard ReAct loop where the tools are **not imported directly** — they are discovered at runtime from an MCP server. The agent connects via `MultiServerMCPClient`, fetches the available tools, and proceeds like any other ReAct agent.

The key difference from the Tavily example is the **client-server split**:

- **`math_server.py`** — a separate process that exposes functions (`add`, `subtract`, `multiply`, `divide`) as MCP tools using `FastMCP`. It has no knowledge of LangChain or LangGraph; it just speaks the MCP protocol over stdio.
- **`math_agent.py`** — the agent. It launches the server as a subprocess, loads its tools at runtime, and builds the graph. You can point `MCP_CONFIG` at any other MCP server (local or HTTP) without changing the agent code.

**Use case:** Connecting an agent to external tool providers (databases, internal APIs, third-party services) through a standardized protocol instead of hard-coding tool imports.  
**Cost:** Same as ReAct — scales with the number of tool calls the LLM makes.

Two examples are included:

| File | Server | Transport | Question |
|---|---|---|---|
| `examples/mcp/math_agent.py` | `math_server.py` (local) | stdio | arithmetic calculation |
| `examples/mcp/docs_agent.py` | `https://docs.langchain.com/mcp` | HTTP | LangGraph documentation search |

**Extra dependency required:**
```bash
pip install langchain-mcp-adapters mcp aiosqlite
```

---

### Reflexion (`examples/reflexion/`)

Draft → tool-call → revise loop with self-critique — the agent reflects on its own output and iterates until satisfied.

---

## Example: Reflexion with Tavily search

```python
from langgraph_compare import *

exp = create_experiment("reflexion_research")
memory = exp.memory

# ... build your Reflexion graph with draft / execute_tools / revise nodes ...

run_multiple_iterations(graph, 1, 10, {"messages": [("user", "Your research question")]})

graph_config = GraphConfig(nodes=["draft", "execute_tools", "revise"])
prepare_data(exp, graph_config)

event_log = load_event_log(exp)
print_analysis(event_log)
generate_artifacts(event_log, graph, exp)
```

---

## Project Structure

```
langgraph_compare/
├── experiment.py       # ExperimentPaths, create_experiment()
├── graph_runner.py     # run_multiple_iterations()
├── sql_to_jsons.py     # export_sqlite_to_jsons()
├── jsons_to_csv.py     # export_jsons_to_csv(), GraphConfig
├── load_events.py      # load_event_log()
├── analyze.py          # global process mining metrics
├── analyze_case_id.py  # per-case process mining metrics
├── visualize.py        # generate_mermaid/prefix_tree/performance_dfg
├── create_report.py    # write_metrics_report, write_sequences_report
├── create_html.py      # compare() — multi-architecture HTML report
├── artifacts.py        # prepare_data(), generate_artifacts()
└── templates/          # Jinja2 HTML report templates
```

---

## License

See [LICENSE](LICENSE).
