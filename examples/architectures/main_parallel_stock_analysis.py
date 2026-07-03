"""
Multi-perspective stock research brief — fan-out / fan-in (map-reduce) pattern.

Why this pattern: the four analyses (bull, bear, macro, technical) are mutually
independent — none needs another's output to run — so they fan out from START
and execute concurrently. A single synthesizer node then joins all four results,
writes a balanced Bulgarian brief, and delivers it to Telegram via curl.
This is the orchestrator-workers / map-reduce shape, not a sequential supervisor
(main_supervision.py) or a peer handoff network (main_network.py).

Universal across tickers: the ticker isn't a separate constant, it's parsed
out of USER_PROMPT below (marked with a leading $, e.g. "$HOOD"), the same
prompt sent to the graph as the human message. Point it at any ticker by
editing that one string — analyst instructions, the Finviz crawl, and the
Telegram labels all derive from it.
"""

import functools
import operator
import re
import subprocess
import os
import sys
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from news_crawler import crawl, get_fundamentals, get_insider_trading

# Load API keys from .env (OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
load_dotenv()

# The single source of truth for what's being analyzed.
if len(sys.argv) < 2:
    sys.exit("Usage: python main_parallel_stock_analysis.py TICKER")
USER_PROMPT = f"Анализирай ${sys.argv[1]} и подготви balanced research brief."

_ticker_match = re.search(r"\$([A-Z]{1,5})", USER_PROMPT)
if not _ticker_match:
    sys.exit(f"Could not find a $TICKER token in USER_PROMPT: {USER_PROMPT!r}")
TICKER = _ticker_match.group(1)

# Finviz news + full article text (via news_crawler.py), replacing Exa as the
# data source. Cached because all four analysts cover the same ticker — no
# reason to re-crawl Finviz and re-fetch every article four times.
# Capped to 8 articles / 800 chars each: article text is often several
# thousand characters, and four parallel agents pulling everything at once is
# exactly what blew past the 128k context limit with Exa's full-page mode.
@functools.lru_cache(maxsize=1)
def _cached_news_digest() -> str:
    articles = crawl(TICKER)[:8]
    return "\n\n".join(
        f"{a['title']} ({a['source']})\n{(a['text'] or '')[:800]}"
        for a in articles
    )


@tool
def search_tool(query: str) -> str:
    """Look up recent news and information about the ticker. Returns a digest of
    the latest Finviz-listed articles (headline, source, and excerpt)."""
    return _cached_news_digest()


# Finviz's valuation/margin/technical snapshot table (P/E, RSI, SMA20/50/200,
# target price, 52-week range, performance windows, etc.) — hard numbers the
# news digest alone can't give the technical/bull/bear analysts. Cached for
# the same reason as the news digest: one Finviz fetch serves all four agents.
@functools.lru_cache(maxsize=1)
def _cached_fundamentals() -> str:
    return "\n".join(f"{k}: {v}" for k, v in get_fundamentals(TICKER).items())


@tool
def fundamentals_tool(query: str = "") -> str:
    """Look up the ticker's key valuation, margin, and technical metrics from
    Finviz: P/E, EPS, margins, RSI, '% Price vs SMA20/50/200' (how far price is
    above/below each moving average — NOT the moving average's price), 52-week
    range, target price, a precomputed 'Target vs Current Price' gap, and
    performance over various windows."""
    return _cached_fundamentals()


# Finviz's insider trading table (who traded, role, date, buy/sell, price,
# shares, resulting stake) — feeds the bear analyst specifically, since a
# cluster of insider sales is a classic bearish signal the news digest and
# fundamentals table don't otherwise surface.
@functools.lru_cache(maxsize=1)
def _cached_insider_trading() -> str:
    records = get_insider_trading(TICKER, limit=15)
    if not records:
        return "No recent insider trading data found."
    return "\n".join(
        f"{r['date']}: {r['insider']} ({r['relationship']}) — {r['transaction']} "
        f"{r['shares']} shares @ {r['cost']}, value {r['value']}, now holds {r['shares_total']}"
        for r in records
    )


@tool
def insider_trading_tool(query: str = "") -> str:
    """Look up the ticker's most recent insider trading activity from Finviz:
    who traded (name and role), date, buy/sell, price, share count, value, and
    their resulting total stake. A cluster of insider sales is a bearish signal."""
    return _cached_insider_trading()


# The macro analyst needs actual Fed/rate-environment data, not ticker news —
# search_tool and fundamentals_tool are both scoped to the ticker and gave it
# nothing to work with, so it was filling its answer with duplicated bull-style
# company content instead of Fed/FOMC/rate context. A real web search fixes
# that; DuckDuckGo needs no API key.
_ddg_search = DuckDuckGoSearchRun()


@tool
def macro_search_tool(query: str) -> str:
    """Search the web for macroeconomic news: Fed/FOMC decisions, interest
    rates, inflation, and their effect on markets. Not ticker-specific — use
    queries like 'Federal Reserve interest rate decision retail trading'."""
    return _ddg_search.invoke(query)


llm = ChatOpenAI(model="gpt-4o")


class AnalysisState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ticker: str
    bull: str
    bear: str
    macro: str
    technical: str
    brief: str


def make_analyst(role_key: str, instructions: str, tools: list | None = None):
    """Factory for a single fan-out worker node.

    Each worker is an isolated react agent with web search; it only writes to
    its own state key (role_key), so the four workers never conflict when
    LangGraph merges their concurrent updates. `tools` defaults to the
    ticker-scoped news+fundamentals tools; pass an explicit list to override
    (e.g. the macro analyst needs real web search, not ticker data).
    """
    agent = create_react_agent(
        llm,
        tools=tools if tools is not None else [search_tool, fundamentals_tool],
        prompt=(
            f"You are a financial analyst covering the stock ticker {TICKER}. "
            f"{instructions} "
            "Respond in Bulgarian, in exactly 2-3 sentences. Be concrete and cite what you found."
        ),
    )

    def node(state: AnalysisState) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        text = result["messages"][-1].content
        send_telegram(f"{ROLE_LABELS[role_key]}:\n{text}")
        return {
            role_key: text,
            "messages": [HumanMessage(content=text, name=role_key)],
        }

    return node


# Bulgarian labels shown as message headers in Telegram, keyed by state field.
ROLE_LABELS = {
    "bull": "🐂 Bull анализ",
    "bear": "🐻 Bear анализ",
    "macro": "🏦 Макро анализ",
    "technical": "📈 Технически анализ",
}

bull_node = make_analyst(
    "bull",
    "Търси catalysts, растежни метрики и положителни analyst ъпгрейди от последните 24 часа. "
    "Цитирай КОНКРЕТНИ analyst firms и техните price targets от search_tool (напр. 'BTIG вдигна "
    "таргета до $X'), не общи фрази като 'анализаторите са позитивни'. Ако споменеш Recom числото "
    "от fundamentals_tool, поясни скалата (1.0=Strong Buy, 5.0=Sell) — не го подавай голо.",
)
bear_node = make_analyst(
    "bear",
    "Отвори директно с най-съществения риск или негативен сигнал — не започвай с bullish контекст "
    "(напр. ръст на акциите), дори да го споменаваш по-късно за баланс. Търси рискове, сигнали за "
    "overvaluation и негативни развития от последните 24 часа. Провери insider_trading_tool — "
    "клъстер от insider продажби е важен bearish сигнал. Провери и fundamentals_tool за 'Target vs "
    "Current Price' — ако таргетът е под текущата цена, това е ограничен upside и трябва да се "
    "спомене изрично, не мимоходом.",
    tools=[search_tool, fundamentals_tool, insider_trading_tool],
)
macro_node = make_analyst(
    "macro",
    "Ти НЕ анализираш компанията или нейните продукти — това е работа на другите анализатори. "
    "Направи поне два отделни macro_search_tool searches: (1) текущия Fed funds rate диапазон и "
    "последното FOMC решение, (2) най-новия FOMC dot plot / summary of economic projections и "
    "датата на СЛЕДВАЩОТО FOMC заседание. Тонът на dot plot-а (по-ястребов/по-гълъбов спрямо "
    "предходния) е също толкова важен, колкото текущата ставка — не representирай средата само "
    "като 'стабилна', ако dot plot-ът сигнализира промяна на очакванията. Обясни как всичко това "
    "влияе на retail trading сектора.",
    tools=[macro_search_tool],
)
technical_node = make_analyst(
    "technical",
    "Провери ценовото движение, обема и ключовите технически нива (RSI, SMA20/50/200, "
    "52-седмичен диапазон, target price) — използвай fundamentals_tool за конкретни цифри. "
    "Провери изрично полето 'Target vs Current Price': ако текущата цена е НАД target price, "
    "това е логическо несъответствие с bullish технически сигнали (ограничен upside) и трябва да "
    "се отбележи директно, не само в обобщение.",
)


def send_telegram(text: str) -> None:
    """Deliver the final brief via the Telegram Bot API using curl.

    Credentials come from env vars, never interpolated into a shell string,
    so there's no command-injection surface even though the brief text is
    LLM-generated.
    """
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    subprocess.run(
        [
            "curl", "-s", "-X", "POST",
            f"https://api.telegram.org/bot{token}/sendMessage",
            "-d", f"chat_id={chat_id}",
            "--data-urlencode", f"text={text}",
        ],
        check=True,
    )


def synthesizer_node(state: AnalysisState) -> dict:
    """Fan-in join: combines all four analyses into one balanced brief, then ships it."""
    synth_prompt = (
        f"Синтезирай следните четири анализа за акцията {TICKER} в единен, балансиран "
        "brief от 5-6 изречения на български. Включи гледна точка от всяка от четирите "
        "перспективи без да звучи едностранчиво.\n\n"
        f"Bull анализ: {state['bull']}\n\n"
        f"Bear анализ: {state['bear']}\n\n"
        f"Макро анализ: {state['macro']}\n\n"
        f"Технически анализ: {state['technical']}"
    )
    brief = llm.invoke(synth_prompt).content
    send_telegram(f"📋 Обобщен brief за {TICKER}:\n{brief}")
    return {"brief": brief, "messages": [HumanMessage(content=brief, name="synthesizer")]}


workflow = StateGraph(AnalysisState)
workflow.add_node("bull_analyst", bull_node)
workflow.add_node("bear_analyst", bear_node)
workflow.add_node("macro_analyst", macro_node)
workflow.add_node("technical_analyst", technical_node)
workflow.add_node("synthesizer", synthesizer_node)

# Fan-out: all four analysts start concurrently from START.
for worker in ["bull_analyst", "bear_analyst", "macro_analyst", "technical_analyst"]:
    workflow.add_edge(START, worker)
    # Fan-in: synthesizer only runs once every worker has written its state key.
    workflow.add_edge(worker, "synthesizer")

workflow.add_edge("synthesizer", END)

graph = workflow.compile()

user_input = {
    "messages": [HumanMessage(content=USER_PROMPT)],
    "ticker": TICKER,
}

result = graph.invoke(user_input, config={"recursion_limit": 50})
print(result["brief"])
