from pathlib import Path
from typing import Optional
from src.job_agent.core.config import settings
from src.job_agent.graph.state import CandidateProfile, JobApplicationState
from src.job_agent.tools.lead_exporter import export_vetted_lead


def build_dispatch_node(
    profile: Optional[CandidateProfile] = None,
    output_dir: str | Path = "outputs",
):
    loaded_profile = profile or CandidateProfile.from_yaml(settings.profile_path)

    def dispatch_node(state: JobApplicationState) -> dict:
        decision = state.get("human_decision") or "approve"
        if decision == "skip":
            return {
                "dispatch_status": "SKIPPED_BY_HUMAN",
                "audit_log": ["[Dispatch] Lead skipped by human reviewer."],
            }

        parsed_jd = state["parsed_jd"]
        draft = state["outreach_draft"]
        discovered_emails = state.get("discovered_emails") or []

        target_email = (
            draft.recipient_email
            or (discovered_emails[0] if discovered_emails else "careers@company.com")
        )

        lead_row = {
            "company_name": parsed_jd.company_name if parsed_jd else "Unknown",
            "role_title": parsed_jd.role_title if parsed_jd else "Software Engineer",
            "location": parsed_jd.location if parsed_jd else "India",
            "tech_match_score": f"{(state.get('tech_match_score') or 0.0):.0%}",
            "matched_skills": ", ".join(state.get("matched_skills") or []),
            "matched_experiences": ", ".join(
                parsed_jd.matched_experience_ids if parsed_jd else []
            ),
            "matched_projects": ", ".join(
                parsed_jd.matched_project_ids if parsed_jd else []
            ),
            "recipient_role": draft.recipient_role_or_name,
            "target_email": target_email,
            "email_subject": draft.email_subject,
            "email_body": draft.email_body,
            "linkedin_note": draft.linkedin_note,
            "static_resume_path": str(
                loaded_profile.candidate.get(
                    "static_resume_path", "assets/AzariushHussainResume.pdf"
                )
            ),
            "job_url": state.get("job_url", ""),
        }

        status, csv_path, xlsx_path = export_vetted_lead(
            lead_row=lead_row,
            output_dir=output_dir,
        )

        return {
            "dispatch_status": status,
            "audit_log": [
                f"[Dispatch] Status={status} -> {xlsx_path.name} & {csv_path.name} (Target: {target_email})"
            ],
        }

    return dispatch_node
