import datetime
import json

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool

from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

from typing import Annotated
from typing_extensions import TypedDict

from pydantic import ValidationError, BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

from dotenv import load_dotenv

from langgraph_compare import *

import shutil, os
_exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "programming_100")
if os.path.exists(_exp_dir):
    shutil.rmtree(_exp_dir)

exp = create_experiment("programming_100")
memory = exp.memory

load_dotenv()

# TOOLS
llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatGroq(model="llama-3.1-8b-instant")
# llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(search=search, max_results=5)


# INITIAL RESPONDER
class CodeAnalysis(BaseModel):
    """Analysis of the proposed solution"""
    technical_gaps: str = Field(description="Identify technical gaps or potential issues in the implementation")
    optimization: str = Field(description="Suggestions for optimization and best practices")
    security: str = Field(description="Security considerations and potential vulnerabilities")


class ProgrammingAnswer(BaseModel):
    """Provide a detailed programming solution with implementation details and considerations."""

    solution: str = Field(
        description="Detailed solution including code examples, implementation details, and explanation"
    )
    analysis: CodeAnalysis = Field(description="Technical analysis of the solution")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching technical improvements or alternative approaches"
    )


class RevisedProgrammingAnswer(ProgrammingAnswer):
    """Provide an improved programming solution with references to documentation and best practices."""

    references: list[str] = Field(
        description="Links to relevant documentation, GitHub repos, or technical resources"
    )


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state["messages"]}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                # Create new messages list with error response
                new_messages = state["messages"] + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                                + json.dumps(self.validator.model_json_schema())
                                + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
                state = {"messages": new_messages}
        return {"messages": response}


# Define system prompts
actor_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert software engineer and programming mentor.
        Current time: {time}

        Your role is to:
        1. {first_instruction}
        2. Analyze the technical aspects of your solution including potential issues, optimizations, and security considerations
        3. Research additional technical resources and documentation to improve the solution

        Focus on:
        - Writing clean, maintainable code
        - Following programming best practices
        - Providing detailed explanations of the implementation
        - Considering edge cases and error handling
        - Suggesting optimizations and improvements""",
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "user",
        "\n\n<system>Review the programming question and provide a solution using the {function_name} function.</system>",
    ),
]).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# Setup initial answer chain
initial_answer_chain = (
        actor_prompt_template.partial(
            first_instruction="Provide a detailed programming solution with implementation details.",
            function_name=ProgrammingAnswer.__name__,
        ) | llm.bind_tools(tools=[ProgrammingAnswer])
)

validator = PydanticToolsParser(tools=[ProgrammingAnswer])
first_responder = ResponderWithRetries(runnable=initial_answer_chain, validator=validator)

# Setup revision chain
revise_instructions = """Improve your programming solution using the additional research:
    - Update the implementation based on best practices and documentation
    - Add error handling and edge cases
    - Optimize the code for better performance
    - Include links to relevant documentation and resources
    - Ensure the solution follows security best practices
    - Add example usage and test cases
"""

revision_chain = (
        actor_prompt_template.partial(
            first_instruction=revise_instructions,
            function_name=RevisedProgrammingAnswer.__name__,
        ) | llm.bind_tools(tools=[RevisedProgrammingAnswer])
)

revision_validator = PydanticToolsParser(tools=[RevisedProgrammingAnswer])
revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)


# TOOL NODE
def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode([
    StructuredTool.from_function(run_queries, name=ProgrammingAnswer.__name__),
    StructuredTool.from_function(run_queries, name=RevisedProgrammingAnswer.__name__),
])


# CONSTRUCT GRAPH
class State(TypedDict):
    messages: Annotated[list, add_messages]


MAX_ITERATIONS = 5
builder = StateGraph(State)

builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def get_num_iterations(state: list):
    """Count the number of iterations in the current state."""
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i


def event_loop(state: dict):
    num_iterations = get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile(checkpointer=memory)

# Example usage
user_input = {"messages": [("user", "How do I implement a secure REST API with rate limiting in Python?")]}

print()
run_multiple_iterations(graph, 1, 5, user_input)
print()

graph_config = GraphConfig(
    nodes=["draft", "execute_tools", "revise"]
)

prepare_data(exp, graph_config)

# ANALYSIS
print()
event_log = load_event_log(exp)
print_analysis(event_log)
print()

generate_artifacts(event_log, graph, exp)