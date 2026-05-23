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

# Reflexion workflow (Shinn et al. 2023):
#   START → draft → execute_tools → revise → (loop back or END)
#
#   1. draft        – LLM produces an initial answer + self-critique + search queries
#   2. execute_tools – Tavily runs the search queries and returns real web results
#   3. revise       – LLM improves the answer using the search results and its own critique
#   4. event_loop   – repeats steps 2-3 up to MAX_ITERATIONS times, then ends
#
# The key insight: forcing structured self-reflection (missing / superfluous) before
# searching means the LLM targets its own blind spots rather than searching at random.

import shutil, os

_exp_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments",
    "climate_100",
)
if os.path.exists(_exp_dir):
    shutil.rmtree(_exp_dir)

exp = create_experiment("climate_100")
memory = exp.memory

load_dotenv()

# ── STEP 0: shared tools ──────────────────────────────────────────────────────
# Tavily is the external evaluator: it grounds the LLM's self-critique in real
# web results, closing the feedback loop between reflection and revision.
llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatGroq(model="llama-3.1-8b-instant")
# llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(search=search, max_results=5)


# ── STEP 1: structured output schemas ────────────────────────────────────────
# Pydantic schemas act as the "episodic memory" from the paper: the LLM is
# forced to commit its critique and search intent to structured fields, which
# are then carried forward into the next revision cycle.
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


# ReviseAnswer extends AnswerQuestion with a required references field, forcing
# the LLM to cite sources after searching — encourages grounded, verifiable answers.
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,
    cite your reflection with references, and finally
    add search queries to improve the answer."""

    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )


# ── STEP 2: actor with retries ────────────────────────────────────────────────
# ResponderWithRetries wraps any LLM chain. If the model's output fails Pydantic
# validation (wrong schema), the error + schema are injected as a ToolMessage and
# the model is given another chance — up to 3 attempts total.
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
                # Feed the validation error back so the model can self-correct.
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


# Shared prompt template used by both the drafter and the revisor.
# {first_instruction} and {function_name} are swapped out per role.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<system>Reflect on the user's original question and the actions taken thus far. Respond using the {function_name} function.</reminder>",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# draft node: produces the first answer + self-critique + search queries
initial_answer_chain = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer.",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion])

validator = PydanticToolsParser(tools=[AnswerQuestion])

first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator
)

# ── STEP 3: revisor ───────────────────────────────────────────────────────────
# revise node: same prompt template, but now the conversation history includes
# Tavily search results, so the LLM can ground its revised answer in real sources.
revise_instructions = """Revise your previous answer using the new information.
        - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions,
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer])

revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)


# ── STEP 4: tool execution node ───────────────────────────────────────────────
# execute_tools node: runs all search queries the LLM produced in bulk.
# Both AnswerQuestion and ReviseAnswer tool calls are routed here via name aliases.
def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)


# ── STEP 5: graph wiring ──────────────────────────────────────────────────────
# The graph encodes the Reflexion loop:
#   draft → execute_tools → revise → (back to execute_tools, or END)
class State(TypedDict):
    messages: Annotated[list, add_messages]


MAX_ITERATIONS = 5  # cap total draft+revise cycles to avoid runaway costs
builder = StateGraph(State)

builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def _get_num_iterations(messages: list):
    # Count consecutive tool/ai messages from the end — each draft+search+revise
    # cycle adds 2 AI messages and 1 tool message, so this tracks loop depth.
    i = 0
    for m in messages[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i


def event_loop(state: dict):
    # After each revision, decide whether to search again or stop.
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile(checkpointer=memory)

# ── RUN ───────────────────────────────────────────────────────────────────────
user_input = {"messages": [("user", "How should we handle the climate crisis?")]}

print()
run_multiple_iterations(graph, 1, 5, user_input)
print()

graph_config = GraphConfig(nodes=["draft", "execute_tools", "revise"])

prepare_data(exp, graph_config)

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
print()
event_log = load_event_log(exp)
print_analysis(event_log)
print()

generate_artifacts(event_log, graph, exp)
