from typing import Optional, Any
from src.job_agent.core.config import settings
from src.job_agent.core.llm_factory import get_structured_llm
from src.job_agent.graph.state import (
    CandidateProfile,
    JobApplicationState,
    ParsedJD,
)
from src.job_agent.harness.validators import (
    evaluate_jd_constraints,
    rank_experiences_and_projects,
)


def build_gatekeeper_node(
    profile: Optional[CandidateProfile] = None,
    structured_llm: Optional[Any] = None,
):
    """LangGraph Gatekeeper node with deterministic tech-stack overlap & experience matching."""
    loaded_profile = profile or CandidateProfile.from_yaml(settings.profile_path)

    exp_descriptions = "\n".join(
        f"- {e.exp_id}: {e.role} at {e.company} ({', '.join(e.tech_stack)})"
        for e in loaded_profile.work_experience
    )
    project_descriptions = "\n".join(
        f"- {p.project_id}: {p.name} ({', '.join(p.tech_stack)})"
        for p in loaded_profile.projects
    )

    def gatekeeper_node(state: JobApplicationState) -> dict:
        llm = structured_llm or get_structured_llm(ParsedJD, role="fast")

        messages = [
            (
                "system",
                "You are a precise technical recruiting parser. Extract structured JD requirements.\n"
                "1. Classify `role_category` accurately (e.g., 'backend', 'full-stack', 'ai_engineer', 'qa_testing', 'devops_sre', 'other').\n"
                "2. Extract all `must_have_skills` and `nice_to_have_skills` as individual technology names.\n"
                "3. Set `strict_btech_be_only = True` ONLY if the JD explicitly mandates B.Tech/B.E. and excludes B.Sc/BCA.\n"
                f"4. Candidate Work Experiences:\n{exp_descriptions}\n"
                f"5. Candidate Projects:\n{project_descriptions}",
            ),
            (
                "human",
                f"Job URL: {state.get('job_url', '')}\n\nRaw JD:\n{state['raw_jd_text']}",
            ),
        ]

        parsed_jd: ParsedJD = llm.invoke(messages)

        # Deterministically rank and assign matching work experiences and projects
        matched_exp_ids, matched_proj_ids = rank_experiences_and_projects(
            parsed_jd, loaded_profile
        )
        parsed_jd.matched_experience_ids = matched_exp_ids
        parsed_jd.matched_project_ids = matched_proj_ids

        # Run full constraint + tech stack overlap check
        (
            is_qualified,
            reason,
            tech_match_score,
            matched_skills,
            missing_skills,
        ) = evaluate_jd_constraints(parsed_jd, loaded_profile)

        log_msg = (
            f"[Gatekeeper] {parsed_jd.company_name} ({parsed_jd.role_title}) -> "
            f"Qualified={is_qualified} | TechMatch={tech_match_score:.0%} "
            f"| MatchedExp={matched_exp_ids} | MatchedProj={matched_proj_ids}"
            + (f" | Reason: {reason}" if reason else "")
        )

        return {
            "parsed_jd": parsed_jd,
            "is_qualified": is_qualified,
            "disqualification_reason": reason,
            "tech_match_score": tech_match_score,
            "matched_skills": matched_skills,
            "missing_must_have_skills": missing_skills,
            "audit_log": [log_msg],
        }

    return gatekeeper_node
