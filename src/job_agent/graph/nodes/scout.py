from typing import Optional, Callable, List
from src.job_agent.graph.state import JobApplicationState
from src.job_agent.tools.scraper import scrape_job_description


def build_scout_node(
    http_fetcher: Optional[Callable[[str], str]] = None,
    mx_resolver: Optional[Callable[[str], List[str]]] = None,
    smtp_prober: Optional[Callable[[str, str], str]] = None,
):
    """
    LangGraph Scout Node:
    Scrapes the job URL (or uses raw_jd_text if provided) and verifies discovered emails
    via DNS MX and SMTP probing.
    """

    def scout_node(state: JobApplicationState) -> dict:
        job_url = state.get("job_url", "")
        raw_text = state.get("raw_jd_text", "")

        result = scrape_job_description(
            job_url=job_url,
            existing_raw_text=raw_text,
            http_fetcher=http_fetcher,
            mx_resolver=mx_resolver,
            smtp_prober=smtp_prober,
        )

        if not result.success:
            return {
                "scrape_error": result.error_message,
                "discovered_emails": [],
                "is_qualified": False,
                "disqualification_reason": f"Scout scrape failed: {result.error_message}",
                "audit_log": [f"[Scout] FAILED ({result.source_tier}): {result.error_message}"],
            }

        statuses = [f"{r.email}:{r.status}" for r in result.verification_report]
        return {
            "raw_jd_text": result.text,
            "discovered_emails": result.discovered_emails,
            "scrape_error": None,
            "audit_log": [
                f"[Scout] Ingested JD via {result.source_tier} ({len(result.text)} chars, "
                f"verified_emails={result.discovered_emails}, checks={statuses})"
            ],
        }

    return scout_node
