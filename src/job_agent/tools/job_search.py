from typing import List
import httpx


def _clean_keywords(raw_query: str) -> str:
    """Strips site: operators, quotes, and boolean ORs that break search engines."""
    tokens = []
    for token in raw_query.split():
        if token.lower().startswith("site:") or token.upper() == "OR":
            continue
        cleaned = token.strip("\"'")
        if cleaned:
            tokens.append(cleaned)
    return " ".join(tokens) or "Python Backend Engineer"


def _search_via_ddgs(keywords: str, max_results: int) -> List[str]:
    """Tier 1: Search via ddgs using clean queries."""
    urls: List[str] = []
    queries = [
        f"{keywords} job apply",
        f"{keywords} careers greenhouse OR lever",
    ]
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            for q in queries:
                for row in ddgs.text(q, max_results=max_results * 2):
                    href = row.get("href", "")
                    if href.startswith(("http://", "https://")) and href not in urls:
                        urls.append(href)
                        if len(urls) >= max_results:
                            return urls
    except Exception:
        pass
    return urls


def _search_via_public_job_apis(keywords: str, max_results: int) -> List[str]:
    """
    Tier 2 Fallback: Queries public, zero-auth JSON tech job APIs (Remotive & Arbeitnow)
    so job discovery always returns live URLs even if search engines rate-limit your IP.
    """
    urls: List[str] = []
    kw_list = [
        k.lower()
        for k in keywords.split()
        if k.lower() not in ("bangalore", "india", "remote")
    ]
    primary_kw = kw_list[0] if kw_list else "python"

    headers = {"User-Agent": "JobAgentScout/1.0"}
    with httpx.Client(timeout=12.0, follow_redirects=True, headers=headers) as client:
        try:
            resp = client.get(
                "https://remotive.com/api/remote-jobs",
                params={
                    "category": "software-dev",
                    "search": primary_kw,
                    "limit": max_results * 3,
                },
            )
            if resp.status_code == 200:
                for job in resp.json().get("jobs", []):
                    url = job.get("url", "")
                    if url and url not in urls:
                        urls.append(url)
                        if len(urls) >= max_results:
                            return urls
        except Exception:
            pass

        try:
            resp = client.get("https://www.arbeitnow.com/api/job-board-api")
            if resp.status_code == 200:
                for job in resp.json().get("data", []):
                    text_blob = f"{job.get('title', '')} {' '.join(job.get('tags', []))}".lower()
                    if any(k in text_blob for k in kw_list) or "python" in text_blob:
                        url = job.get("url", "")
                        if url and url not in urls:
                            urls.append(url)
                            if len(urls) >= max_results:
                                return urls
        except Exception:
            pass

    return urls


def discover_job_urls(
    query: str = "Python FastAPI Bangalore",
    max_results: int = 3,
) -> List[str]:
    """Discovers live job URLs using DDGS first, falling back to public JSON job APIs."""
    keywords = _clean_keywords(query)
    urls = _search_via_ddgs(keywords, max_results=max_results)
    if len(urls) < max_results:
        for fallback_url in _search_via_public_job_apis(
            keywords, max_results=max_results
        ):
            if fallback_url not in urls:
                urls.append(fallback_url)
            if len(urls) >= max_results:
                break
    return urls[:max_results]
