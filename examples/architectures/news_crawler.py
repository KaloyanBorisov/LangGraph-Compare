"""
Crawls a ticker's Finviz quote page for news articles and the fundamentals/
technical snapshot table.

- get_news_links() / crawl(): pull every headline out of the news table
  (id="news-table") and extract each linked article's body text with
  trafilatura (handles boilerplate/nav/ad stripping far better than raw
  BeautifulSoup text extraction). Sites that block bots, redirect to a
  paywall, or aren't plain HTML (e.g. a YouTube link showing up in the feed)
  are skipped with a note rather than raising.
- get_fundamentals(): pull the ~80-metric valuation/margin/technical snapshot
  table (P/E, RSI, SMA, target price, performance windows, etc.) into a flat
  dict.
- get_insider_trading(): pull the insider trading table (who traded, role,
  date, buy/sell, price, shares, value, resulting stake) into a list of
  records — signals like a cluster of insider sales feed the bear analyst.

All three read from the same quote page, so the raw HTML fetch is cached per
ticker (functools.lru_cache) to avoid hitting Finviz three times per run.
"""

import functools
import sys

import requests
import trafilatura
from bs4 import BeautifulSoup

FINVIZ_URL = "https://finviz.com/quote.ashx?t={ticker}"

# Finviz returns a bot-block page without a convincing browser User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


@functools.lru_cache(maxsize=8)
def _fetch_quote_page(ticker: str) -> BeautifulSoup:
    resp = requests.get(FINVIZ_URL.format(ticker=ticker), headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_news_links(ticker: str) -> list[dict]:
    """Scrape the Finviz news table for a ticker: [{title, url, source}, ...]."""
    soup = _fetch_quote_page(ticker)
    table = soup.find(id="news-table")
    if table is None:
        raise RuntimeError(f"No news table found for ticker {ticker!r} — page layout may have changed.")

    links = []
    for row in table.find_all("tr"):
        anchor = row.find("a")
        if anchor is None or not anchor.get("href"):
            continue
        source_span = row.find("span")
        source = source_span.get_text(strip=True).strip("()") if source_span else None
        links.append({
            "title": anchor.get_text(strip=True),
            "url": anchor["href"],
            "source": source,
        })
    return links


# Finviz's own labels are ambiguous or actively misleading once flattened to
# text — most notably "SMA20/50/200" are NOT the moving-average price, they're
# how far the current price is above/below that SMA. Relabeled here so
# consumers (including the LLM analysts) don't misread them as prices.
_LABEL_FIXUPS = {
    "SMA20": "% Price vs SMA20 (20-day moving average)",
    "SMA50": "% Price vs SMA50 (50-day moving average)",
    "SMA200": "% Price vs SMA200 (200-day moving average)",
    "Recom": "Recom (analyst recommendation, 1.0=Strong Buy .. 5.0=Sell)",
}


def get_fundamentals(ticker: str) -> dict:
    """Scrape Finviz's valuation/margin/technical snapshot table into a flat dict.

    The page renders the ~80-metric grid as six separate <table class="snapshot-table2">
    elements (one per visual column), each a flat sequence of label/value <td>
    pairs — they're merged here into one dict. Ambiguous labels (SMA20/50/200,
    Recom) are rewritten via _LABEL_FIXUPS; cells with a nested delta (e.g.
    "52W High" = value + <small> % from that high) get a space inserted so
    "153.86" and "-26.95%" don't get glued into "153.86-26.95%".
    """
    soup = _fetch_quote_page(ticker)
    tables = soup.select("table.snapshot-table2")
    if not tables:
        raise RuntimeError(f"No snapshot table found for ticker {ticker!r} — page layout may have changed.")

    metrics = {}
    for table in tables:
        cells = [td.get_text(separator=" ", strip=True) for td in table.find_all("td")]
        for label, value in zip(cells[0::2], cells[1::2]):
            metrics[_LABEL_FIXUPS.get(label, label)] = value

    if "Price" in metrics and "Target Price" in metrics:
        try:
            price, target = float(metrics["Price"]), float(metrics["Target Price"])
            pct = (target - price) / price * 100
            metrics["Target vs Current Price"] = (
                f"target is {pct:+.1f}% {'above' if pct >= 0 else 'below'} current price "
                f"({'upside' if pct >= 0 else 'downside/limited upside'})"
            )
        except ValueError:
            pass

    return metrics


INSIDER_COLUMNS = [
    "insider", "relationship", "date", "transaction",
    "cost", "shares", "value", "shares_total", "sec_form4",
]


def get_insider_trading(ticker: str, limit: int = 20) -> list[dict]:
    """Scrape Finviz's insider trading table for a ticker, most recent first.

    Each record: insider name, relationship (e.g. "10% Owner", "CEO"), date,
    transaction (Sale/Buy/Option Exercise), cost per share, #shares, value ($),
    resulting total shares held, and the SEC Form 4 filing link.
    """
    soup = _fetch_quote_page(ticker)
    table = soup.select_one("table.body-table")
    if table is None:
        raise RuntimeError(f"No insider trading table found for ticker {ticker!r} — page layout may have changed.")

    records = []
    for row in table.find_all("tr", class_="fv-insider-row")[:limit]:
        cells = row.find_all("td")
        record = dict(zip(INSIDER_COLUMNS, (td.get_text(strip=True) for td in cells)))
        if record:
            records.append(record)
    return records


def extract_article_text(url: str) -> str | None:
    """Fetch a URL and return its extracted main-body text, or None if extraction fails."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None
    return trafilatura.extract(downloaded, include_comments=False, include_tables=False)


def crawl(ticker: str) -> list[dict]:
    """Full pipeline: Finviz news links -> extracted article text for each."""
    results = []
    for link in get_news_links(ticker):
        text = extract_article_text(link["url"])
        results.append({**link, "text": text})
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python news_crawler.py TICKER")
    ticker = sys.argv[1]

    print(f"{'=' * 80}\nFUNDAMENTALS/TECHNICAL SNAPSHOT\n{'=' * 80}")
    for label, value in get_fundamentals(ticker).items():
        print(f"{label:<20} {value}")

    print(f"\n{'=' * 80}\nINSIDER TRADING\n{'=' * 80}")
    for record in get_insider_trading(ticker):
        print(record)

    for i, article in enumerate(crawl(ticker), 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}] {article['title']}  ({article['source']})")
        print(article["url"])
        print("-" * 80)
        if article["text"]:
            print(article["text"][:2000])
        else:
            print("[extraction failed or unsupported content type — skipped]")
