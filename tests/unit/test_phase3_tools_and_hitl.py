from pathlib import Path
from typing import Any, Callable
import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.job_agent.core.config import settings
from src.job_agent.graph.state import (
    CandidateProfile,
    CriticEvaluation,
    OutreachDraftPayload,
    ParsedJD,
)
from src.job_agent.graph.workflow import compile_job_application_graph
from src.job_agent.tools.scraper import (
    scrape_job_description,
    extract_and_verify_emails,
)
from src.job_agent.tools.lead_exporter import export_vetted_lead


class MockStructuredLLM:
    def __init__(self, handler: Callable[[Any], Any]):
        self._handler = handler

    def invoke(self, messages: Any) -> Any:
        return self._handler(messages)


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile.from_yaml(settings.profile_path)


def test_scraper_falls_back_to_html_and_verifies_emails():
    def fake_fetcher(url: str) -> str:
        if "r.jina.ai" in url:
            raise RuntimeError("Simulated Jina Reader outage")
        return """
        <html>
            <body>
                <h1>Software Engineer - Backend (Python / FastAPI)</h1>
                <p>We are hiring Backend Engineers in Bangalore with 1-2 years experience.
                Send your profile to hiring@finstack.io or bounced@finstack.io or dead@nomxdomain.io.</p>
            </body>
        </html>
        """

    def fake_mx_resolver(domain: str) -> list[str]:
        if domain == "finstack.io":
            return ["mx1.finstack.io"]
        return []

    def fake_smtp_prober(email: str, mx_host: str) -> str:
        if email == "hiring@finstack.io":
            return "smtp_verified"
        if email == "bounced@finstack.io":
            return "smtp_rejected"
        return "smtp_unreachable"

    res = scrape_job_description(
        job_url="https://finstack.io/careers/backend",
        existing_raw_text="",
        http_fetcher=fake_fetcher,
        mx_resolver=fake_mx_resolver,
        smtp_prober=fake_smtp_prober,
    )
    assert res.success is True
    assert res.source_tier == "direct_html"
    assert res.discovered_emails == ["hiring@finstack.io"]


def test_inferred_emails_are_dropped_unless_smtp_verified():
    emails_dropped, reports_dropped = extract_and_verify_emails(
        text="We are hiring a Python Backend Engineer in Bangalore.",
        job_url="https://stealthstartup.ai/jobs/1",
        infer_if_missing=True,
        mx_resolver=lambda d: ["mx.stealthstartup.ai"],
        smtp_prober=lambda email, mx: "catch_all",
    )
    assert emails_dropped == []
    assert all(r.status == "unverified_inferred_dropped" for r in reports_dropped)

    ats_emails, ats_reports = extract_and_verify_emails(
        text="We are hiring a Python Backend Engineer in Bangalore.",
        job_url="https://boards.greenhouse.io/acme/jobs/123",
        infer_if_missing=True,
        mx_resolver=lambda d: ["mx.greenhouse.io"],
        smtp_prober=lambda email, mx: "smtp_verified",
    )
    assert ats_emails == []
    assert ats_reports == []

    emails_kept, _ = extract_and_verify_emails(
        text="We are hiring a Python Backend Engineer in Bangalore.",
        job_url="https://verifiedstartup.io/jobs/1",
        infer_if_missing=True,
        mx_resolver=lambda d: ["mx.verifiedstartup.io"],
        smtp_prober=lambda email, mx: (
            "smtp_verified" if email.startswith("careers@") else "smtp_rejected"
        ),
    )
    assert emails_kept == ["careers@verifiedstartup.io"]


def test_lead_exporter_enforces_idempotency_across_csv_and_xlsx(tmp_path: Path):
    sample_row = {
        "company_name": "Nexusolve",
        "role_title": "Full-Stack Engineer",
        "location": "Bangalore",
        "tech_match_score": "100%",
        "matched_skills": "Python, FastAPI, React.js, PostgreSQL",
        "matched_experiences": "exp_lyfverse, exp_vibtree",
        "matched_projects": "proj_quickchat",
        "recipient_role": "Engineering Lead",
        "target_email": "careers@nexusolve.io",
        "email_subject": "Full-Stack Engineer — Python, FastAPI & React.js",
        "email_body": "Hi Team, sharing my resume for the Full-Stack role.",
        "linkedin_note": "Hi! Applied for the Full-Stack Engineer role.",
        "static_resume_path": "assets/AzariushHussainResume.pdf",
        "job_url": "https://nexusolve.io/jobs/1",
    }

    status1, csv_path, xlsx_path = export_vetted_lead(sample_row, output_dir=tmp_path)
    assert status1 == "EXPORTED"
    assert csv_path.exists()
    assert xlsx_path.exists()

    status2, _, _ = export_vetted_lead(sample_row, output_dir=tmp_path)
    assert status2 == "DUPLICATE_SKIPPED"


def test_full_graph_hitl_pause_and_resume_exports_lead(
    profile: CandidateProfile, tmp_path: Path
):
    mock_gatekeeper_llm = MockStructuredLLM(
        lambda _: ParsedJD(
            company_name="HyperScaleAI",
            role_title="Backend & AI Engineer",
            location="Bangalore",
            role_category="backend",
            min_required_yoe=1.0,
            strict_btech_be_only=False,
            role_level="entry",
            must_have_skills=["Python", "FastAPI", "PostgreSQL", "Redis", "RAG"],
        )
    )

    mock_writer_llm = MockStructuredLLM(
        lambda _: OutreachDraftPayload(
            recipient_role_or_name="CTO",
            recipient_email=None,
            email_subject="Backend & AI Engineer — Python, FastAPI, Redis & RAG",
            email_body=(
                "Hi Team, saw your Backend & AI Engineer role. At Lyfverse Technologies, "
                "I built FastAPI/Redis services (95+ Core Web Vitals), integrated 4 payment gateways, "
                "and cut Cloudflare image costs by 100% ($2,900+/yr saved). Attached my resume!"
            ),
            linkedin_note=(
                "Hi! Saw the Backend & AI Engineer role. Built FastAPI, Redis & RAG pipelines at Lyfverse "
                "(100% image cost reduction). Sent my resume via email!"
            ),
            claimed_skills=["Python", "FastAPI", "Redis", "RAG"],
            referenced_experience_ids=["exp_lyfverse"],
            referenced_project_ids=["proj_quickchat"],
            included_urls=["https://github.com/AzariushHussain"],
        )
    )

    mock_critic_llm = MockStructuredLLM(
        lambda _: CriticEvaluation(
            passed=True,
            grounding_score=0.98,
            specificity_score=0.95,
            revision_instructions=None,
        )
    )

    checkpointer = MemorySaver()
    graph = compile_job_application_graph(
        profile=profile,
        gatekeeper_llm=mock_gatekeeper_llm,
        writer_llm=mock_writer_llm,
        critic_llm=mock_critic_llm,
        mx_resolver=lambda d: ["mx.hyperscaleai.io"],
        smtp_prober=lambda email, mx: "smtp_verified",
        output_dir=tmp_path,
        checkpointer=checkpointer,
        enable_hitl=True,
    )

    config = {"configurable": {"thread_id": "thread_hitl_hyperscale"}}

    paused_state = graph.invoke(
        {
            "job_url": "https://hyperscaleai.io/jobs/backend",
            "raw_jd_text": (
                "Hiring Backend & AI Engineer in Bangalore (1-2 yrs). Stack: Python, FastAPI, "
                "PostgreSQL, Redis, RAG. Contact: cto@hyperscaleai.io for direct applications."
            ),
            "revision_count": 0,
            "audit_log": [],
        },
        config=config,
    )

    snapshot = graph.get_state(config)
    assert snapshot.next == ("dispatch_lead",)
    assert paused_state.get("dispatch_status") is None
    assert "cto@hyperscaleai.io" in paused_state["discovered_emails"]

    graph.update_state(config, {"human_decision": "approve"})
    final_state = graph.invoke(None, config=config)

    assert final_state["dispatch_status"] == "EXPORTED"
    assert (tmp_path / "vetted_job_leads.xlsx").exists()
    assert (tmp_path / "vetted_job_leads.csv").exists()
