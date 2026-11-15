from pathlib import Path
import pytest
from src.job_agent.graph.state import (
    CandidateProfile,
    ParsedJD,
    OutreachDraftPayload,
)
from src.job_agent.harness.validators import (
    evaluate_jd_constraints,
    rank_experiences_and_projects,
    validate_outreach_draft,
)

PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/job_agent/memory/candidate_profile.yaml"
)


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile.from_yaml(PROFILE_PATH)


def test_gatekeeper_accepts_matching_python_backend_role(profile: CandidateProfile):
    jd = ParsedJD(
        company_name="NexusAI",
        role_title="Software Engineer - Backend & AI",
        location="Bangalore",
        role_category="backend",
        min_required_yoe=1.0,
        strict_btech_be_only=False,
        role_level="entry",
        must_have_skills=["Python", "FastAPI", "Django", "PostgreSQL", "RAG"],
    )
    is_qualified, reason, score, matched, missing = evaluate_jd_constraints(jd, profile)
    assert is_qualified is True
    assert reason is None
    assert score == 1.0
    assert len(matched) == 5
    assert len(missing) == 0


def test_gatekeeper_rejects_mismatched_java_dotnet_tech_stack(profile: CandidateProfile):
    java_jd = ParsedJD(
        company_name="EnterpriseBank",
        role_title="Associate Software Engineer",
        location="Bangalore",
        role_category="backend",
        min_required_yoe=1.0,
        strict_btech_be_only=False,
        role_level="entry",
        must_have_skills=["Java", "Spring Boot", "C#", ".NET", "Oracle DB"],
    )
    is_qualified, reason, score, matched, missing = evaluate_jd_constraints(java_jd, profile)
    assert is_qualified is False
    assert score == 0.0
    assert "Zero overlap with core anchor stack" in reason


def test_gatekeeper_ranks_vibtree_and_lyfverse_for_rag_serverless_jd(profile: CandidateProfile):
    rag_jd = ParsedJD(
        company_name="AgenticLabs",
        role_title="AI Backend Engineer",
        location="Bangalore",
        role_category="ai_engineer",
        min_required_yoe=1.0,
        must_have_skills=["Python", "Django", "AWS Lambda", "AWS SQS", "LangChain", "RAG", "LLaMA 2"],
        nice_to_have_skills=["WebRTC", "Socket.IO"],
    )
    matched_exps, matched_projs = rank_experiences_and_projects(rag_jd, profile)
    # Vibtree has Django, AWS Lambda, AWS SQS, LangChain, RAG, LLaMA 2!
    assert matched_exps[0] == "exp_vibtree"
    assert "exp_lyfverse" in matched_exps
    # Nice-to-have WebRTC + Socket.IO should rank proj_webrtc #1
    assert matched_projs[0] == "proj_webrtc"


def test_outreach_validator_catches_unapproved_skills_and_bad_urls(profile: CandidateProfile):
    bad_draft = OutreachDraftPayload(
        recipient_role_or_name="CTO",
        recipient_email="cto@nexusai.io",
        email_subject="Software Engineer — Python / FastAPI",
        email_body="Hi, I built systems and deployed them on Kubernetes with Golang.",
        linkedin_note="Hi! Applying for the Backend role.",
        claimed_skills=["FastAPI", "Kubernetes", "Golang"],
        referenced_experience_ids=["exp_lyfverse"],
        referenced_project_ids=["proj_quickchat", "proj_fake"],
        included_urls=["https://github.com/AzariushHussain/nonexistent-repo"],
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
        referenced_experience_ids=["exp_lyfverse", "exp_vibtree"],
        referenced_project_ids=["proj_quickchat"],
        included_urls=["https://github.com/AzariushHussain"],
    )

    report = validate_outreach_draft(clean_draft, profile)
    assert report.passed is True
    assert len(report.violations) == 0
