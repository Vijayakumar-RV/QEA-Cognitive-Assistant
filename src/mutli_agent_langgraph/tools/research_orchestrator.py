
from typing import List, Dict
from src.mutli_agent_langgraph.tools.research_tools import tavily_web_search, serpapi_search, extract_webpage_content, dedupe_and_rank

def research_pipeline(query: str, model, max_sources: int = 6) -> Dict:
    # 1) search
    t = tavily_web_search.invoke({"query": query})
    s = serpapi_search.invoke({"query": query})
    sources = dedupe_and_rank(t, s)[:max_sources]

    # 2) fetch/clean
    corpus=[]
    for i, src in enumerate(sources, start=1):
        url = src.get("url","")
        text = extract_webpage_content.invoke({"url": url})
        if not text: 
            continue
        corpus.append({"id": i, "title": src["title"], "url": url, "snippet": src.get("snippet",""), "text": text})

    # 3) map summaries
    bullets=[]
    for c in corpus:
        prompt = f"Summarize the key facts from this article in 3-5 bullets, no fluff:\n\n{c['text'][:5000]}"
        resp = model.invoke(prompt).content
        bullets.append((c["id"], resp.strip()))

    # 4) reduce + citations
    numbered = "\n".join([f"[{cid}] {b}" for cid,b in bullets])
    final_prompt = f"""Combine the following source summaries into 6–10 precise bullets.
Add inline numeric citations like [1], [3] where appropriate.
Be factual; if unsure, omit.

Query: {query}

Source Summaries:
{numbered}
"""
    final = model.invoke(final_prompt).content

    # final markdown
    md = "## Summary\n" + final + "\n\n## Sources\n" + "\n".join([f"[{c['id']}] {c['title']} — {c['url']}" for c in corpus])
    return {"markdown": md, "sources": corpus, "bullets": bullets}
