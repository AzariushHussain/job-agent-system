import re
from pathlib import Path
import yaml
import pytest

from src.job_agent.core.config import settings
from src.job_agent.graph.state import (
    CandidateProfile,
    CriticEvaluation,
    OutreachDraftPayload,
    ParsedJD,
)
from src.job_agent.graph.workflow import (
    compile_job_application_graph,
    route_after_gatekeeper,
)
from src.job_agent.harness.validators import (
    evaluate_jd_constraints,
    rank_experiences_and_projects,
    validate_outreach_draft,
)

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_jds.yaml"
NUMERIC_CLAIM_REGEX = re.compile(r"\b\d+(?:\.\d+)?%|\$\d[\d,]*\+?\b")


class _EvalMockLLM:
    def __init__(self, handler):
        self._handler = handler

    def invoke(self, messages):
        return self._handler(messages)


def verify_prose_metrics_grounded(email_body: str, profile: CandidateProfile) -> bool:
    """Checks that every % or $ metric in the email prose exists in candidate_profile.yaml."""
    evidence_text = " ".join(
        " ".join(e.base_bullets + e.locked_metrics) for e in profile.work_experience
    ).lower()
    for claim in NUMERIC_CLAIM_REGEX.findall(email_body):
        if claim.lower() not in evidence_text:
            return False
    return True


@pytest.fixture(scope="module")
def profile() -> CandidateProfile:
    return CandidateProfile.from_yaml(settings.profile_path)


@pytest.fixture(scope="module")
def golden_cases() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["cases"]


# 1. WORKFLOW METRIC 1: Gatekeeper Qualification & Conditional Routing Accuracy (100% Gate)
def test_1_gatekeeper_qualification_and_routing_accuracy(
    profile: CandidateProfile, golden_cases: list[dict]
):
    matches = 0
    for case in golden_cases:
        jd = ParsedJD(**{
            k: case[k]
            for k in (
                "company_name",
                "role_title",
                "location",
                "role_category",
                "role_level",
                "min_required_yoe",
                "strict_btech_be_only",
                "must_have_skills",
                "nice_to_have_skills",
            )
        })
        is_qualified, reason, score, _, _ = evaluate_jd_constraints(jd, profile)
        next_node = route_after_gatekeeper({"is_qualified": is_qualified})

        if (
            is_qualified == case["expected_qualified"]
            and next_node == case["expected_next_node"]
        ):
            matches += 1
        else:
            pytest.fail(
                f"[{case['case_id']}] Expected qualified={case['expected_qualified']} "
                f"(route={case['expected_next_node']}), got qualified={is_qualified} "
                f"(route={next_node}, reason={reason}, score={score})"
            )

    accuracy = matches / len(golden_cases)
    assert accuracy == 1.0


# 2. WORKFLOW METRIC 2: Experience & Project Matcher Ranking Accuracy (100% Gate)
def test_2_experience_and_project_ranking_accuracy(
    profile: CandidateProfile, golden_cases: list[dict]
):
    checks = 0
    correct = 0
    for case in golden_cases:
        jd = ParsedJD(**{
            k: case[k]
            for k in (
                "company_name",
                "role_title",
                "location",
                "role_category",
                "role_level",
                "min_required_yoe",
                "strict_btech_be_only",
                "must_have_skills",
                "nice_to_have_skills",
            )
        })
        matched_exps, matched_projs = rank_experiences_and_projects(jd, profile)

        if "expected_top_experience" in case:
            checks += 1
            if matched_exps and matched_exps[0] == case["expected_top_experience"]:
                correct += 1

        if "expected_top_project" in case:
            checks += 1
            if matched_projs and matched_projs[0] == case["expected_top_project"]:
                correct += 1

    assert checks > 0 and (correct / checks) == 1.0


# 3. WORKFLOW METRIC 3: End-to-End Fact Groundedness & Hallucination Guardrail (100% Gate)
def test_3_outreach_fact_groundedness_and_metric_entailment(profile: CandidateProfile):
    # Valid draft citing verified Lyfverse (100%, $2,900+) and Vibtree (50%) metrics
    valid_draft = OutreachDraftPayload(
        recipient_role_or_name="Hiring Team",
        recipient_email="careers@finflow.io",
        email_subject="Backend Engineer — Python, FastAPI & Redis",
        email_body=(
            "Hi Team, at Lyfverse Technologies I cut Cloudflare image costs by 100% "
            "(saving $2,900+ annually) and at Vibtree I boosted engagement by 50% "
            "using Python, FastAPI, Django, and Redis."
        ),
        linkedin_note="Hi! Applying for the Backend Engineer role.",
        claimed_skills=["Python", "FastAPI", "Django", "Redis"],
        referenced_experience_ids=["exp_lyfverse", "exp_vibtree"],
        referenced_project_ids=["proj_quickchat"],
        included_urls=["https://github.com/AzariushHussain"],
    )
    assert validate_outreach_draft(valid_draft, profile).passed is True
    assert verify_prose_metrics_grounded(valid_draft.email_body, profile) is True

    # Hallucinated metric draft (claims 85% and $50,000+ not in resume)
    hallucinated_metric_draft = valid_draft.model_copy(
        update={
            "email_body": "Hi Team, I reduced AWS costs by 85% saving $50,000+ annually."
        }
    )
    assert (
        verify_prose_metrics_grounded(hallucinated_metric_draft.email_body, profile)
        is False
    )


# 4. WORKFLOW METRIC 4: Writer-Critic Loop Convergence & Bounded Retry Efficiency
def test_4_writer_critic_loop_convergence_within_max_revisions(
    profile: CandidateProfile, tmp_path: Path
):
    attempts = {"count": 0}

    def writer_with_first_attempt_violation(_msgs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            # Attempt 1 fails harness due to unapproved skill 'Rust'
            return OutreachDraftPayload(
                recipient_role_or_name="CTO",
                recipient_email="cto@finflow.io",
                email_subject="Backend Engineer",
                email_body="Built microservices in Python and Rust.",
                linkedin_note="Hi! Applying for the Backend role.",
                claimed_skills=["Python", "Rust"],
                referenced_experience_ids=["exp_lyfverse"],
                referenced_project_ids=["proj_quickchat"],
                included_urls=["https://github.com/AzariushHussain"],
            )
        # Attempt 2 self-corrects using compact harness feedback
        return OutreachDraftPayload(
            recipient_role_or_name="CTO",
            recipient_email="cto@finflow.io",
            email_subject="Backend Engineer — Python, FastAPI & Redis",
            email_body=(
                "Hi Team, at Lyfverse Technologies I cut image costs by 100% ($2,900+ saved) "
                "and integrated 4 payment gateways using Python, FastAPI, and Redis."
            ),
            linkedin_note="Hi! Applying for the Backend role.",
            claimed_skills=["Python", "FastAPI", "Redis"],
            referenced_experience_ids=["exp_lyfverse"],
            referenced_project_ids=["proj_quickchat"],
            included_urls=["https://github.com/AzariushHussain"],
        )

    graph = compile_job_application_graph(
        profile=profile,
        gatekeeper_llm=_EvalMockLLM(
            lambda _: ParsedJD(
                company_name="FinFlow India",
                role_title="Software Engineer - Backend",
                location="Bangalore",
                role_category="backend",
                role_level="entry",
                min_required_yoe=1.0,
                strict_btech_be_only=False,
                must_have_skills=["Python", "FastAPI", "Redis"],
            )
        ),
        writer_llm=_EvalMockLLM(writer_with_first_attempt_violation),
        critic_llm=_EvalMockLLM(
            lambda _: CriticEvaluation(
                passed=True, grounding_score=0.98, specificity_score=0.95
            )
        ),
        mx_resolver=lambda _: ["mx.finflow.io"],
        smtp_prober=lambda e, m: "smtp_verified",
        output_dir=tmp_path,
    )

    final_state = graph.invoke(
        {
            "job_url": "https://finflow.io/jobs/backend",
            "raw_jd_text": "Hiring Backend Engineer in Bangalore with Python, FastAPI, Redis. Contact cto@finflow.io.",
            "revision_count": 0,
            "audit_log": [],
        }
    )

    assert attempts["count"] == 2
    assert final_state["revision_count"] <= settings.max_revisions
    assert final_state["harness_report"].passed is True
    assert final_state["critic_eval"].passed is True
    assert final_state["dispatch_status"] == "EXPORTED"
