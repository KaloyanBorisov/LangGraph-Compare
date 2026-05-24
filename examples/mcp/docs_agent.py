"""
MCP example: a LangGraph ReAct agent that searches LangChain/LangGraph
documentation via the official docs MCP server at https://docs.langchain.com/mcp.

Architecture:
  START → agent → (tools → agent)* → END

Tools are discovered at runtime from the remote MCP server — no local tool
imports needed. The agent calls the docs search tools, reads the results,
and synthesises an answer grounded in the official documentation.
"""

import asyncio
import copy
import shutil
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph_compare import (
    create_experiment,
    GraphConfig,
    prepare_data,
    load_event_log,
    print_analysis,
    generate_artifacts,
)

load_dotenv()

# ── Experiment setup ──────────────────────────────────────────────────────────
_exp_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments",
    "mcp_docs_100",
)
if os.path.exists(_exp_dir):
    shutil.rmtree(_exp_dir)

exp = create_experiment("mcp_docs_100")

llm = ChatOpenAI(model="gpt-4o-mini")

# ── MCP server config ─────────────────────────────────────────────────────────
# Connects to the official LangChain docs MCP server over HTTP.
# Tools (e.g. search_docs_by_lang_chain) are discovered at runtime.
MCP_CONFIG = {
    "docs": {
        "url": "https://docs.langchain.com/mcp",
        "transport": "streamable_http",
    }
}

USER_INPUT = {"messages": [("user", "How do I use create_react_agent in LangGraph?")]}


# MCP tools are async-only, so graph.stream (sync) raises NotImplementedError.
# This mirrors run_multiple_iterations using astream instead.
async def run_multiple_iterations_async(graph, starting_thread_id, num_repetitions, user_input_template, recursion_limit=100):
    for i in range(num_repetitions):
        current_thread_id = str(starting_thread_id + i)
        config = {"configurable": {"thread_id": current_thread_id}, "recursion_limit": recursion_limit}
        user_input = copy.deepcopy(user_input_template)

        print("#" * 30)
        print(f"Iteration: {i + 1}, Thread_ID {current_thread_id}")
        print("#" * 30)

        step_num = 0
        async for event in graph.astream(user_input, config, stream_mode="values"):
            for key, value in event.items():
                if "__end__" not in value:
                    print(f"Step {step_num}:")
                    print(value)
                    print("---")
            step_num += 1


async def main():
    # Fetch available tools from the remote docs MCP server at startup.
    # The server exposes search tools; the agent decides which ones to call.
    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()

    # AsyncSqliteSaver is required because MCP tools are async-only and the
    # graph runs via astream — SqliteSaver raises NotImplementedError in async context.
    async with AsyncSqliteSaver.from_conn_string(exp.database) as checkpointer:
        # create_react_agent wires: agent node ↔ ToolNode in a standard ReAct loop.
        graph = create_react_agent(llm, tools, checkpointer=checkpointer)

        print()
        await run_multiple_iterations_async(graph, 1, 5, USER_INPUT)
        print()

        # Extract event log from SQLite and transform to CSV for analysis.
        graph_config = GraphConfig(nodes=["agent", "tools"])
        prepare_data(exp, graph_config)

        # ── Analysis ──────────────────────────────────────────────────────────────
        print()
        event_log = load_event_log(exp)
        print_analysis(event_log)
        print()

        generate_artifacts(event_log, graph, exp)


if __name__ == "__main__":
    asyncio.run(main())
