from typing import Optional, Any
from src.job_agent.core.config import settings
from src.job_agent.core.llm_factory import get_structured_llm
from src.job_agent.graph.state import (
    CandidateProfile,
    JobApplicationState,
    OutreachDraftPayload,
)


def build_outreach_writer_node(
    profile: Optional[CandidateProfile] = None,
    structured_llm: Optional[Any] = None,
):
    """LangGraph Outreach Writer node prioritized by Gatekeeper's matched experiences and projects."""
    loaded_profile = profile or CandidateProfile.from_yaml(settings.profile_path)

    def outreach_writer_node(state: JobApplicationState) -> dict:
        llm = structured_llm or get_structured_llm(OutreachDraftPayload, role="writer")

        parsed_jd = state["parsed_jd"]
        matched_exp_ids = set(parsed_jd.matched_experience_ids) if parsed_jd else set()
        matched_proj_ids = set(parsed_jd.matched_project_ids) if parsed_jd else set()

        # Prioritize matched work experiences
        prioritized_exps = [
            e for e in loaded_profile.work_experience if e.exp_id in matched_exp_ids
        ] or loaded_profile.work_experience[:2]

        experience_context = "\n".join(
            f"- {e.exp_id} ({e.role} at {e.company}, {e.start_date}–{e.end_date}): "
            f"{' '.join(e.base_bullets)} [Tech: {', '.join(e.tech_stack)}]"
            for e in prioritized_exps
        )

        prioritized_projects = [
            p for p in loaded_profile.projects if p.project_id in matched_proj_ids
        ] or loaded_profile.projects[:1]

        matched_projects_context = "\n".join(
            f"- {p.project_id} ({p.name}): {' '.join(p.base_bullets)} | Repo: {p.repo_url}"
            for p in prioritized_projects
        )

        feedback_items = []
        harness_report = state.get("harness_report")
        if harness_report and not harness_report.passed:
            feedback_items.extend(
                [f"HARNESS VIOLATION: {v}" for v in harness_report.violations]
            )
        critic_eval = state.get("critic_eval")
        if critic_eval and critic_eval.revision_instructions:
            feedback_items.append(f"CRITIC FEEDBACK: {critic_eval.revision_instructions}")

        feedback_section = (
            "\n".join(feedback_items)
            if feedback_items
            else "First draft — no prior violations."
        )

        candidate_name = loaded_profile.candidate.get("name", "Azariush Hussain")
        allowed_skills = ", ".join(sorted(loaded_profile.flat_allowed_skills))
        valid_exp_ids = ", ".join(sorted(loaded_profile.valid_experience_ids))
        valid_project_ids = ", ".join(sorted(loaded_profile.valid_project_ids))
        valid_urls = ", ".join(sorted(loaded_profile.valid_urls))
        matched_skills_str = ", ".join(state.get("matched_skills") or [])

        messages = [
            (
                "system",
                f"You are an elite technical cold-outreach writer acting on behalf of {candidate_name}.\n"
                "Write a concise cold email (<= 180 words) and LinkedIn note (<= 400 characters).\n\n"
                "STRICT HARNESS CONSTRAINTS:\n"
                f"1. ALLOWED SKILLS ONLY: {allowed_skills}\n"
                f"2. VERIFIED EXPERIENCE IDS ONLY: {valid_exp_ids}\n"
                f"3. VERIFIED PROJECT IDS ONLY: {valid_project_ids}\n"
                f"4. VERIFIED URLS ONLY: {valid_urls}\n\n"
                f"TOP MATCHED WORK EXPERIENCES FOR THIS JD:\n{experience_context}\n\n"
                f"TOP MATCHED PROJECTS FOR THIS JD:\n{matched_projects_context}\n\n"
                f"DIRECT SKILL OVERLAP WITH JD: {matched_skills_str}\n\n"
                f"REVISION FEEDBACK FROM PRIOR ATTEMPT:\n{feedback_section}",
            ),
            (
                "human",
                f"Company: {parsed_jd.company_name if parsed_jd else 'Target Company'}\n"
                f"Role: {parsed_jd.role_title if parsed_jd else 'Software Engineer'}\n"
                f"Must-Have Skills: {', '.join(parsed_jd.must_have_skills) if parsed_jd else ''}\n"
                f"Raw JD:\n{state['raw_jd_text']}",
            ),
        ]

        draft: OutreachDraftPayload = llm.invoke(messages)
        attempt_num = state.get("revision_count", 0) + 1

        return {
            "outreach_draft": draft,
            "audit_log": [
                f"[OutreachWriter] Generated draft attempt #{attempt_num} citing exp={draft.referenced_experience_ids}, proj={draft.referenced_project_ids}"
            ],
        }

    return outreach_writer_node
