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
_exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "math_100")
if os.path.exists(_exp_dir):
    shutil.rmtree(_exp_dir)

exp = create_experiment("math_100")
memory = exp.memory

load_dotenv()

# TOOLS
llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatGroq(model="llama-3.1-8b-instant")
# llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(search=search, max_results=5)


# INITIAL RESPONDER
class MathematicalAnalysis(BaseModel):
    """Analysis of the mathematical solution"""
    assumptions: str = Field(description="Key assumptions and constraints in the problem")
    verification: str = Field(description="Methods to verify the solution's correctness")
    alternative_approaches: str = Field(description="Other possible solution methods")


class MathSolution(BaseModel):
    """Provide a detailed mathematical solution with step-by-step reasoning."""

    solution: str = Field(
        description="Step-by-step solution including equations, calculations, and explanations"
    )
    analysis: MathematicalAnalysis = Field(description="Mathematical analysis of the solution")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching mathematical concepts or alternative approaches"
    )


class RevisedMathSolution(MathSolution):
    """Provide an improved mathematical solution with references to theorems and proofs."""

    references: list[str] = Field(
        description="Links to mathematical resources, proofs, or relevant academic papers"
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
        """You are an expert mathematician and mathematical educator.
        Current time: {time}

        Your role is to:
        1. {first_instruction}
        2. Analyze the mathematical approach, including assumptions and verification methods
        3. Research additional mathematical resources and proofs to enhance the solution

        Focus on:
        - Providing clear, step-by-step solutions
        - Explaining mathematical concepts thoroughly
        - Using proper mathematical notation
        - Verifying solutions with different methods
        - Considering edge cases and special conditions
        - Citing relevant theorems and proofs""",
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "user",
        "\n\n<system>Review the mathematical problem and provide a solution using the {function_name} function.</system>",
    ),
]).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# Setup initial answer chain
initial_answer_chain = (
        actor_prompt_template.partial(
            first_instruction="Provide a detailed step-by-step mathematical solution.",
            function_name=MathSolution.__name__,
        ) | llm.bind_tools(tools=[MathSolution])
)

validator = PydanticToolsParser(tools=[MathSolution])
first_responder = ResponderWithRetries(runnable=initial_answer_chain, validator=validator)

# Setup revision chain
revise_instructions = """Improve your mathematical solution using the additional research:
    - Incorporate relevant theorems and proofs
    - Add alternative solution methods
    - Verify the solution's correctness
    - Include references to mathematical resources
    - Consider special cases and constraints
    - Add numerical examples where appropriate
"""

revision_chain = (
        actor_prompt_template.partial(
            first_instruction=revise_instructions,
            function_name=RevisedMathSolution.__name__,
        ) | llm.bind_tools(tools=[RevisedMathSolution])
)

revision_validator = PydanticToolsParser(tools=[RevisedMathSolution])
revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)

# TOOL NODE
def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode([
    StructuredTool.from_function(run_queries, name=MathSolution.__name__),
    StructuredTool.from_function(run_queries, name=RevisedMathSolution.__name__),
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
user_input = {
    "messages": [("user", "How do I solve a system of linear equations with 3 variables using elimination method?")]}

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