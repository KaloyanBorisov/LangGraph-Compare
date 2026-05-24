:orphan:

.. _advanced_examples:

Advanced Examples
#################

.. contents:: Table of Contents

Preface
*******
As advanced examples, we are going to use the following tutorials from LangGraph:

* `Agent Architectures - Multi-Agent Network <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/>`_
* `Agent Architectures - Multi-Agent Supervisor <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/>`_
* `Agent Architectures - Hierarchical Agent Teams <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/>`_

We are also not going to focus on a basic usage - the main focus is going to be on :class:`langgraph_compare.jsons_to_csv.GraphConfig`.

If You want to check basic usage, refer to: :ref:`getting_started`.

Multi-Agent Network
*******************
This example is based on `Agent Architectures - Multi-Agent Network <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/>`_ and expands on :ref:`exporting_jsons_to_csv` from :ref:`getting_started`.

.. figure:: img/network.png
  :width: 800

  `Multi-Agent Network diagram - LangGraph Documentation <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/>`_


The premise of this example is to show that :code:`GraphConfig` can have multiple nodes.

In case of this example - we will have 2 nodes - :code:`researcher`, :code:`chart_generator`.

**Example:**

.. code-block:: python

    # Needed imports
    from langgraph_compare.experiment import create_experiment
    from langgraph_compare.jsons_to_csv import GraphConfig, export_jsons_to_csv

    # Init for experiment project structure
    exp = create_experiment("test")

    # We can add multiple nodes!
    graph_config = GraphConfig(
        nodes=['researcher','chart_generator']
    )

    # You can provide You own file name as an optional attribute csv_path.
    # Otherwise it will use the default file name - "csv_output.csv"
    export_jsons_to_csv(exp, graph_config)

Multi-Agent Supervisor
**********************
This example is based on `Agent Architectures - Multi-Agent Supervisor <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/>`_. It introduces the concept of a :code:`Supervisor` - a node that controls other nodes.

.. figure:: img/supervisor.png
  :width: 800

  `Multi-Agent Supervisor diagram - LangGraph Documentation <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/>`_

In this example, we will introduce :class:`langgraph_compare.jsons_to_csv.SupervisorConfig`. It will supervise the graph - working more or less work the same as :code:`GraphConfig`. The concept of supervisors will make more sense in :ref:`hierarchical_agent_teams`.

**Example:**

.. code-block:: python

    # Needed imports
    from langgraph_compare.experiment import create_experiment
    from langgraph_compare.jsons_to_csv import GraphConfig, SupervisorConfig, export_jsons_to_csv

    # Init for experiment project structure
    exp = create_experiment("test")

    # Supervisor for graph
    supervisor = SupervisorConfig(
        name="supervisor",
        supervisor_type="graph"
    )

    # Config with supervisor and additional nodes
    graph_config = GraphConfig(
        supervisors=[supervisor],
        nodes=["Researcher", "Coder"]
    )

    # You can provide You own file name as an optional attribute csv_path.
    # Otherwise it will use the default file name - "csv_output.csv"
    export_jsons_to_csv(exp, graph_config)

.. _hierarchical_agent_teams:

Hierarchical Agent Teams
************************
This example is based on `Agent Architectures - Hierarchical Agent Teams <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/>`_. It introduces the concept of a :code:`SubgraphConfig` - a node that controls other nodes.

.. figure:: img/hierarchical.png
  :width: 800

  `Hierarchical Agent Teams diagram - LangGraph Documentation <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/>`_

In this example, we have a :code:`Graph` that is build from two :code:`SubGraphs`. Those graphs are controlled by a :code:`Supervisor` - that routes traffic to subgraphs.
Furthermore, every graph has its own supervisor - that controls what is happening inside of it.

IMPORTANT: Be sure to call supervisors with different names - so you can differentiate between them! Calling supervisors with the same names WILL brake the parser.

**Example:**

.. code-block:: python

    # Needed imports
    from langgraph_compare.experiment import create_experiment
    from langgraph_compare.jsons_to_csv import GraphConfig, SubgraphConfig, SupervisorConfig, export_jsons_to_csv

    # Init for experiment project structure
    exp = create_experiment("test")

    # Config for entire graph supervisor
    graph_supervisor = SupervisorConfig(
        name="graph_supervisor",
        supervisor_type="graph"
    )

    # Config for Research Team subgraph supervisor
    research_supervisor = SupervisorConfig(
        name="research_supervisor",
        supervisor_type="subgraph"
    )

    # Config for Paper Writing Team subgraph supervisor
    paper_supervisor = SupervisorConfig(
        name="paper_supervisor",
        supervisor_type="subgraph"
    )

    # Config for Research Team subgraph
    research_team = SubgraphConfig(
        name="ResearchTeam",
        nodes=["Search", "WebScraper"],
        supervisor=research_supervisor
    )

    # Config for Paper Writing Team subgraph
    paper_team = SubgraphConfig(
        name="PaperWritingTeam",
        nodes=["DocWriter", "NoteTaker","ChartGenerator"],
        supervisor=paper_supervisor
    )

    # Config for complete graph
    graph_config = GraphConfig(
        supervisors=[graph_supervisor],
        subgraphs=[research_team, paper_supervisor]
    )

    # You can provide You own file name as an optional attribute csv_path.
    # Otherwise it will use the default file name - "csv_output.csv"
    export_jsons_to_csv(exp, graph_config)

Notice how:

* every supervisor has a config (both graph and subgraphs) - but they have a different :code:`supervisor_type`.
* every graph has a config (both graph and subgraphs) - but they are using different classes: :code:`GraphConfig` or :code:`SubgraphConfig`
* :code:`GraphConfig` doesn't have :code:`nodes` defined - since they are being taken care of by subgraphs.

MCP — Model Context Protocol
*****************************
These examples show how to connect a LangGraph ReAct agent to tools exposed via the
`Model Context Protocol (MCP) <https://modelcontextprotocol.io/>`_ instead of importing tools directly.
The agent discovers available tools at runtime from an MCP server — the graph code does not change
regardless of which server is used.

Both examples use the same two-node ReAct loop:

.. code-block:: text

    START → agent ⇄ tools → END

Because MCP tools are async-only, the graph must run via :code:`astream` instead of :code:`stream`,
and :code:`AsyncSqliteSaver` must be used in place of the default :code:`SqliteSaver`.

**Extra dependencies required:**

.. code-block:: bash

    pip install langchain-mcp-adapters mcp aiosqlite

MCP — Local Math Server (stdio)
================================
:code:`examples/mcp/math_agent.py` connects to a local MCP server (:code:`math_server.py`) that
exposes arithmetic tools (:code:`add`, :code:`subtract`, :code:`multiply`, :code:`divide`) via the
:code:`stdio` transport. The client launches the server as a subprocess automatically.

**Example:**

.. code-block:: python

    import asyncio
    from langchain_openai import ChatOpenAI
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph_compare import create_experiment, GraphConfig, prepare_data, load_event_log, print_analysis, generate_artifacts

    exp = create_experiment("mcp_math")
    llm = ChatOpenAI(model="gpt-4o-mini")

    MCP_CONFIG = {
        "math": {
            "command": "python",
            "args": ["math_server.py"],
            "transport": "stdio",
        }
    }

    async def main():
        client = MultiServerMCPClient(MCP_CONFIG)
        tools = await client.get_tools()

        async with AsyncSqliteSaver.from_conn_string(exp.database) as checkpointer:
            graph = create_react_agent(llm, tools, checkpointer=checkpointer)

            # ... run iterations, prepare_data, load_event_log, print_analysis, generate_artifacts

    asyncio.run(main())

MCP — Remote Docs Server (HTTP)
================================
:code:`examples/mcp/docs_agent.py` connects to the official LangChain documentation MCP server
over HTTP (:code:`streamable_http` transport). No local server process is needed.

**Example:**

.. code-block:: python

    MCP_CONFIG = {
        "docs": {
            "url": "https://docs.langchain.com/mcp",
            "transport": "streamable_http",
        }
    }

The agent uses this config identically to the stdio example — only :code:`MCP_CONFIG` changes.
The graph structure, checkpointer, and analysis pipeline remain the same.