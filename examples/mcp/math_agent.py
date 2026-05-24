"""
MCP example: a LangGraph ReAct agent that calls tools served over MCP.

Architecture:
  START → agent → (tools → agent)* → END

The agent node calls the LLM; if the LLM emits tool calls, ToolNode
executes them via the MCP server. The loop repeats until the LLM
produces a plain message (no tool calls), then the graph ends.

Install the extra dependency before running:
    poetry add langchain-mcp-adapters
or
    pip install langchain-mcp-adapters
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

# MCP tools are async-only, so graph.stream (sync) raises NotImplementedError.
# This async replacement mirrors run_multiple_iterations using astream instead.
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

load_dotenv()

# ── Experiment setup ──────────────────────────────────────────────────────────
_exp_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments",
    "mcp_math_100",
)
if os.path.exists(_exp_dir):
    shutil.rmtree(_exp_dir)

exp = create_experiment("mcp_math_100")

llm = ChatOpenAI(model="gpt-4o-mini")

# ── MCP server config ─────────────────────────────────────────────────────────
# "math" points to a local MCP server that exposes arithmetic tools.
# Swap the args path for any MCP server you want to connect to.
# For an HTTP-based server use: {"url": "http://localhost:8000/mcp", "transport": "streamable_http"}
MCP_CONFIG = {
    "math": {
        "command": "python",
        "args": [os.path.join(os.path.dirname(__file__), "math_server.py")],
        "transport": "stdio",
    }
}

USER_INPUT = {"messages": [("user", "What is (3 + 5) * 12 / 4?")]}


async def main():
    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()

    # AsyncSqliteSaver is required because MCP tools are async-only and the
    # graph runs via astream — SqliteSaver raises NotImplementedError in async context.
    async with AsyncSqliteSaver.from_conn_string(exp.database) as checkpointer:
        graph = create_react_agent(llm, tools, checkpointer=checkpointer)

        print()
        await run_multiple_iterations_async(graph, 1, 5, USER_INPUT)
        print()

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
