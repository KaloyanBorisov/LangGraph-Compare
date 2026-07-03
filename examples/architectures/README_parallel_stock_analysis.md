# Parallel Stock Analysis

`main_parallel_stock_analysis.py` generates a balanced research brief for any
stock ticker and delivers it straight to Telegram.

## Universal ticker

The ticker isn't a hardcoded constant — it's parsed out of `USER_PROMPT`, the
same prompt sent to the graph as the human message (marked with a leading
`$`, e.g. `"Анализирай $HOOD и подготви balanced research brief."`). Analyst
instructions, the Finviz lookups, and the Telegram labels all derive from it.

The ticker is a required CLI argument — there's no silent default:

```bash
python examples/architectures/main_parallel_stock_analysis.py TSLA
```

(the VS Code launch config passes `HOOD` explicitly via `"args"`).

## Pattern: fan-out / fan-in (map-reduce)

The task naturally splits into four independent analyses — none needs another's
output to run:

- **Bull analyst** — catalysts, growth metrics, positive analyst upgrades (last 24h)
- **Bear analyst** — risks, overvaluation signals, insider-selling clusters
- **Macro analyst** — Fed/rate context and its effect on the retail trading sector
- **Technical analyst** — price action, volume, key levels

Because they're independent, all four run **concurrently**, fanning out from
`START`. Each is a `create_react_agent` with its own tool set (see below), and
writes only to its own key in shared state (`bull`, `bear`, `macro`,
`technical`) — so their concurrent writes never collide.

Once all four finish, a single **synthesizer** node fans back in, combines the
four analyses into one balanced 5-6 sentence brief in Bulgarian, and sends it.

This is the same shape as orchestrator-workers / map-reduce: split independent
subtasks across workers, then join. It's distinct from the other examples in
this folder — `main_supervision.py` is a sequential, LLM-routed supervisor, and
`main_network.py` is a peer-to-peer handoff chain — because here there's no
routing decision to make and no dependency between workers.

```
        ┌─► bull_analyst ──────┐
        ├─► bear_analyst ──────┤
START ──┤                      ├─► synthesizer ─► END
        ├─► macro_analyst ─────┤
        └─► technical_analyst ─┘
```

## Data sources (`news_crawler.py`)

Not every analyst gets the same tools — each is scoped to what its role
actually needs. Paid search APIs were tried and dropped first: Tavily's
results were unreliable for this use case, and Exa's default full-page text
blew past the 128k context limit across four parallel agents. Everything now
comes from scraping Finviz's quote page (`https://finviz.com/quote.ashx?t=TICKER`),
whose raw HTML fetch is cached per ticker so the four analysts share one
request instead of hitting Finviz repeatedly:

- **`search_tool`** (bull, bear, technical) — the news table (`#news-table`):
  headline, source, URL, plus the article's body text extracted with
  `trafilatura`. Capped to 8 articles / 800 characters each.
- **`fundamentals_tool`** (bull, bear, technical) — the ~80-metric valuation
  and technical snapshot table (P/E, EPS, margins, RSI, 52-week range, target
  price, performance windows). Two fixes applied here after review:
  - Finviz's own `SMA20/50/200` labels are misleading once flattened to text
    — they're **not** the moving-average price, they're how far the current
    price is above/below that SMA. Relabeled to `% Price vs SMA20 (20-day
    moving average)` etc. so neither the LLM nor a reader misreads them.
  - A `Target vs Current Price` field is precomputed (e.g. "target is -4.9%
    below current price (downside/limited upside)") so analysts can't miss a
    price-above-target situation the way earlier runs did.
- **`insider_trading_tool`** (bear only) — the insider trading table (who
  traded, role, date, buy/sell, price, shares, resulting stake). A cluster of
  insider sales is a bearish signal the news digest and fundamentals table
  don't otherwise surface.
- **`macro_search_tool`** (macro only) — general web search via
  `DuckDuckGoSearchRun` (free, no API key). The macro analyst gets **no**
  ticker-scoped tools at all: giving it `search_tool`/`fundamentals_tool`
  originally caused it to fill its answer with duplicated bull-style company
  news instead of actual Fed/FOMC/rate content, since it had nothing else to
  work with. Its instructions now require at least two separate searches
  (current Fed funds rate + latest FOMC decision, and the FOMC dot plot /
  next meeting date) and explicitly warn against calling the environment
  "stable" if the dot plot signals a shift.

Run the crawler standalone for any ticker (prints fundamentals, insider
trading, and news in one go):

```bash
python examples/architectures/news_crawler.py TSLA
```

### Known limitations

- **Next FOMC meeting date / dot plot specifics**: `DuckDuckGoSearchRun` is
  general snippet search, not a dedicated economic calendar — it can get the
  exact next-meeting date wrong. A proper fix would pull from a real calendar
  source (e.g. Finviz's own `economy.ashx` page) instead.
- **Named analyst upgrades**: the bull analyst is instructed to cite specific
  firms and price targets (not just "analysts are positive"), but the 8-article
  news digest doesn't always contain that level of detail. Finviz has a
  dedicated analyst-ratings-change table that isn't wired in yet.

## Telegram delivery

Each analyst posts its own message to Telegram as soon as it finishes (labeled
🐂/🐻/🏦/📈, so you can tell them apart even though they may arrive out of
order), and the synthesizer posts the final 📋 brief. Delivery is a plain
`curl` call to the Bot API; credentials are read from environment variables and
passed as separate argv entries (never interpolated into a shell string), so
LLM-generated text in the message body can't be used for command injection.

Requires in `.env`:

```
OPENAI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

- `TELEGRAM_BOT_TOKEN`: create a bot via [@BotFather](https://t.me/BotFather) (`/newbot`).
- `TELEGRAM_CHAT_ID`: message your bot once, then call
  `curl https://api.telegram.org/bot<TOKEN>/getUpdates` and read `chat.id` from
  the response.

## Running it

```bash
docker exec langgraph-compare-dev python examples/architectures/main_parallel_stock_analysis.py TICKER
```

or via the "Parallel Stock Analysis" VS Code launch config (runs with `HOOD`).

This is a single `graph.invoke()` call — no experiment folder, checkpointer, or
process-mining pipeline is involved. It's a one-off script, not wired into
`prepare_data` / `generate_artifacts` / the cross-architecture `compare()`
report like the other examples in this folder.
