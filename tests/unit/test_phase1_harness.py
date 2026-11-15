from pathlib import Path
import pytest
from src.job_agent.graph.state import (
    CandidateProfile,
    ParsedJD,
    OutreachDraftPayload,
)
from src.job_agent.harness.validators import (
    evaluate_jd_constraints,
    validate_outreach_draft,
)

PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/job_agent/memory/candidate_profile.yaml"
)


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile.from_yaml(PROFILE_PATH)


def test_gatekeeper_accepts_valid_associate_role(profile: CandidateProfile):
    jd = ParsedJD(
        company_name="NexusAI",
        role_title="Software Engineer - Backend & AI",
        location="Bangalore",
        is_remote=False,
        min_required_yoe=1.0,
        strict_btech_be_only=False,
        role_level="entry",
        must_have_skills=["Python", "FastAPI", "Django", "PostgreSQL", "RAG"],
        matched_project_ids=["proj_quickchat"],
    )
    is_qualified, reason = evaluate_jd_constraints(jd, profile)
    assert is_qualified is True
    assert reason is None


def test_outreach_validator_catches_unapproved_skills_and_bad_urls(profile: CandidateProfile):
    bad_draft = OutreachDraftPayload(
        recipient_role_or_name="CTO",
        recipient_email="cto@nexusai.io",
        email_subject="Software Engineer — Python / FastAPI",
        email_body="Hi, I built systems and deployed them on Kubernetes with Golang.",
        linkedin_note="Hi! Applying for the Backend role.",
        claimed_skills=["FastAPI", "Kubernetes", "Golang"],  # K8s & Golang not in resume allowlist
        referenced_project_ids=["proj_quickchat", "proj_fake"],  # proj_fake is invalid
        included_urls=["https://github.com/AzariushHussain/nonexistent-repo"],  # Unverified URL
    )

    report = validate_outreach_draft(bad_draft, profile)
    assert report.passed is False
    assert len(report.violations) == 4


def test_outreach_validator_passes_clean_draft(profile: CandidateProfile):
    clean_draft = OutreachDraftPayload(
        recipient_role_or_name="Engineering Lead",
        recipient_email="careers@nexusai.io",
        email_subject="Software Engineer (Backend / Full Stack) — Python, FastAPI, Django & RAG",
        email_body=(
            "Hi Team, I saw your opening for a Backend Software Engineer. Over nearly 2 years across "
            "Lyfverse Technologies, Friggly.ai, and Vibtree, I've built scalable FastAPI/Django APIs, "
            "cut Cloudflare image-processing costs by 100% ($2,900+/yr saved), integrated 4 payment "
            "gateways, and shipped RAG features with LangChain and LLaMA 2. I also architected QuickChat "
            "using Redis Pub/Sub and WebSockets. Attached is my resume—would love to connect."
        ),
        linkedin_note=(
            "Hi! Saw the Software Engineer opening. I have ~2 yrs experience with Python, Django, FastAPI, "
            "Redis & RAG (cut cloud image costs 100% & boosted API speed 30-95+ CWV). Sent my resume via email!"
        ),
        claimed_skills=["Python", "FastAPI", "Django", "Redis", "RAG", "LangChain", "WebSockets"],
        referenced_project_ids=["proj_quickchat"],
        included_urls=["https://github.com/AzariushHussain"],
    )

    report = validate_outreach_draft(clean_draft, profile)
    assert report.passed is True
    assert len(report.violations) == 0
