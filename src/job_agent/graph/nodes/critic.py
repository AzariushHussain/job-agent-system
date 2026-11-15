from typing import Optional, Any
from src.job_agent.core.config import settings
from src.job_agent.core.llm_factory import get_structured_llm
from src.job_agent.graph.state import (
    CandidateProfile,
    CriticEvaluation,
    JobApplicationState,
)
from src.job_agent.harness.validators import validate_outreach_draft


def build_critic_node(
    profile: Optional[CandidateProfile] = None,
    structured_llm: Optional[Any] = None,
):
    """Two-stage LangGraph Critic node (Stage 1: Python Harness, Stage 2: LLM Rubric)."""
    loaded_profile = profile or CandidateProfile.from_yaml(settings.profile_path)

    def critic_node(state: JobApplicationState) -> dict:
        draft = state["outreach_draft"]
        current_revisions = state.get("revision_count", 0)

        # STAGE 1: Zero-Token Deterministic Harness Check
        harness_report = validate_outreach_draft(draft, loaded_profile)
        if not harness_report.passed:
            violation_summary = "; ".join(harness_report.violations)
            short_circuit_eval = CriticEvaluation(
                passed=False,
                grounding_score=0.0,
                specificity_score=0.5,
                revision_instructions=f"Fix deterministic harness violations: {violation_summary}",
            )
            return {
                "harness_report": harness_report,
                "critic_eval": short_circuit_eval,
                "revision_count": current_revisions + 1,
                "audit_log": [
                    f"[Critic] FAILED Stage 1 Harness Check ({len(harness_report.violations)} violations): {violation_summary}"
                ],
            }

        # STAGE 2: LLM Rubric Evaluation
        llm = structured_llm or get_structured_llm(CriticEvaluation, role="fast")
        parsed_jd = state.get("parsed_jd")

        messages = [
            (
                "system",
                "You are a strict Cold Outreach Quality Critic. Score grounding_score (0.0-1.0) and specificity_score (0.0-1.0).\n"
                "Set passed = True ONLY if grounding_score >= 0.90 AND specificity_score >= 0.80.",
            ),
            (
                "human",
                f"Target Role: {parsed_jd.role_title if parsed_jd else 'Software Engineer'} at {parsed_jd.company_name if parsed_jd else 'Company'}\n"
                f"Email Subject: {draft.email_subject}\n"
                f"Email Body:\n{draft.email_body}\n\n"
                f"LinkedIn Note:\n{draft.linkedin_note}",
            ),
        ]

        eval_result: CriticEvaluation = llm.invoke(messages)

        passed = (
            eval_result.passed
            and eval_result.grounding_score >= settings.min_grounding_score
            and eval_result.specificity_score >= settings.min_specificity_score
        )
        eval_result.passed = passed
        next_revision_count = current_revisions if passed else current_revisions + 1

        return {
            "harness_report": harness_report,
            "critic_eval": eval_result,
            "revision_count": next_revision_count,
            "audit_log": [
                f"[Critic] Stage 2 LLM Eval -> passed={passed} "
                f"(grounding={eval_result.grounding_score:.2f}, specificity={eval_result.specificity_score:.2f})"
            ],
        }

    return critic_node
