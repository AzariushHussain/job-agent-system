from pathlib import Path
from typing import Callable, Any
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


class MockStructuredLLM:
    def __init__(self, handler: Callable[[Any], Any]):
        self._handler = handler

    def invoke(self, messages: Any) -> Any:
        return self._handler(messages)


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile.from_yaml(settings.profile_path)


def test_gatekeeper_short_circuits_disqualified_jd(
    profile: CandidateProfile, tmp_path: Path
):
    mock_gatekeeper_llm = MockStructuredLLM(
        lambda _: ParsedJD(
            company_name="StrictCorp",
            role_title="Backend Engineer",
            location="Bangalore",
            min_required_yoe=1.0,
            strict_btech_be_only=True,
            role_level="entry",
            must_have_skills=["Python", "Django"],
            matched_project_ids=["proj_quickchat"],
        )
    )

    checkpointer = MemorySaver()
    graph = compile_job_application_graph(
        profile=profile,
        gatekeeper_llm=mock_gatekeeper_llm,
        mx_resolver=lambda _: [],
        output_dir=tmp_path,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "test_thread_disqualified"}}
    result = graph.invoke(
        {
            "job_url": "https://strictcorp.io/jobs/1",
            "raw_jd_text": "B.Tech/B.E. mandatory for this Backend role. B.Sc graduates are not eligible.",
            "revision_count": 0,
            "audit_log": [],
        },
        config=config,
    )

    assert result["is_qualified"] is False
    assert "B.Tech/B.E." in result["disqualification_reason"]
    assert result.get("outreach_draft") is None


def test_writer_critic_loop_self_corrects_on_harness_violation(
    profile: CandidateProfile, tmp_path: Path
):
    mock_gatekeeper_llm = MockStructuredLLM(
        lambda _: ParsedJD(
            company_name="PayScaleAI",
            role_title="Software Engineer (Backend)",
            location="Bangalore",
            min_required_yoe=1.5,
            strict_btech_be_only=False,
            role_level="entry",
            must_have_skills=["Python", "FastAPI", "Redis", "PostgreSQL"],
            matched_project_ids=["proj_quickchat"],
        )
    )

    call_counter = {"count": 0}

    def fake_writer(messages):
        call_counter["count"] += 1
        system_prompt_text = messages[0][1]
        if call_counter["count"] == 1:
            return OutreachDraftPayload(
                recipient_role_or_name="Engineering Manager",
                recipient_email="hiring@payscaleai.io",
                email_subject="Backend Engineer — Python, FastAPI & Kubernetes",
                email_body="Hi, I built FastAPI services and Kubernetes clusters at Lyfverse.",
                linkedin_note="Hi! Applying for the Backend Engineer role.",
                claimed_skills=["Python", "FastAPI", "Kubernetes"],
                referenced_experience_ids=["exp_lyfverse"],
                referenced_project_ids=["proj_quickchat"],
                included_urls=["https://github.com/AzariushHussain"],
            )
        else:
            assert (
                "HARNESS VIOLATION: Unauthorized skill claimed in outreach: 'Kubernetes'"
                in system_prompt_text
            )
            return OutreachDraftPayload(
                recipient_role_or_name="Engineering Manager",
                recipient_email="hiring@payscaleai.io",
                email_subject="Software Engineer — Python, FastAPI, Redis & RAG",
                email_body=(
                    "Hi Team, I'm applying for the Backend Software Engineer role. At Lyfverse Technologies, "
                    "I integrated 4 payment gateways (Razorpay, Cashfree, Dodo, Stripe), cut Cloudflare "
                    "image costs by 100% ($2,900+/yr saved), and optimized SQL/Redis caching for 95+ Core Web Vitals. "
                    "I also built QuickChat using Redis Pub/Sub and WebSockets. Attached my resume!"
                ),
                linkedin_note=(
                    "Hi! Saw the Backend Engineer role. I have ~2 yrs experience in Python, FastAPI, Django & Redis "
                    "(integrated 4 payment gateways & cut image costs 100%). Sent my resume via email!"
                ),
                claimed_skills=["Python", "FastAPI", "Redis", "WebSockets", "SQL"],
                referenced_experience_ids=["exp_lyfverse"],
                referenced_project_ids=["proj_quickchat"],
                included_urls=["https://github.com/AzariushHussain"],
            )

    mock_writer_llm = MockStructuredLLM(fake_writer)

    mock_critic_llm = MockStructuredLLM(
        lambda _: CriticEvaluation(
            passed=True,
            grounding_score=0.98,
            specificity_score=0.92,
            revision_instructions=None,
        )
    )

    checkpointer = MemorySaver()
    graph = compile_job_application_graph(
        profile=profile,
        gatekeeper_llm=mock_gatekeeper_llm,
        writer_llm=mock_writer_llm,
        critic_llm=mock_critic_llm,
        mx_resolver=lambda _: [],
        output_dir=tmp_path,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "test_thread_payscale"}}
    final_state = graph.invoke(
        {
            "job_url": "https://payscaleai.io/careers/backend",
            "raw_jd_text": "Looking for a Backend Engineer with 1-2 years experience in Python, FastAPI, Redis.",
            "revision_count": 0,
            "audit_log": [],
        },
        config=config,
    )

    saved_snapshot = graph.get_state(config)
    assert saved_snapshot.values["critic_eval"].passed is True
    assert call_counter["count"] == 2
    assert "Kubernetes" not in final_state["outreach_draft"].claimed_skills
    assert final_state["dispatch_status"] == "EXPORTED"
