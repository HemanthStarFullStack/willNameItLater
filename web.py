"""Web search tool. Free, no API key (DuckDuckGo), and plain HTTP — so the same
approach ports to the mobile app later. Returns short snippets to keep the
local model's context small.
"""
from ddgs import DDGS


def search(query, k=4):
    """Return up to k results as [{title, url, snippet}]. Never raises."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        hits = DDGS().text(query, max_results=k)
    except Exception:
        return []
    out = []
    for h in hits or []:
        out.append({"title": h.get("title", ""),
                    "url": h.get("href", ""),
                    "snippet": h.get("body", "")})
    return out


def format_results(results):
    """Compact text block to feed the model as tool output."""
    if not results:
        return "No web results found."
    return "\n\n".join(
        f'[{i+1}] {r["title"]}\n{r["snippet"]}\n({r["url"]})'
        for i, r in enumerate(results)
    )
