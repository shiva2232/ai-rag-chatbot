from duckduckgo_search import DDGS

def web_search(query, max_results=5):
    results = []
    
    try:
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                title = r.get("title", "")
                snippet = r.get("body", "")
                link = r.get("href", "")
                
                results.append(f"[{i+1}] {title}\n{snippet}\n{link}")
    
    except Exception as e:
        return f"Web search failed: {str(e)}"
    
    return "\n\n".join(results)