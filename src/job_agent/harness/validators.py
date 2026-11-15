from typing import Tuple, Optional
from src.job_agent.graph.state import (
    CandidateProfile,
    ParsedJD,
    OutreachDraftPayload,
    HarnessCheckResult,
)

MAX_EMAIL_WORDS = 180
MAX_LINKEDIN_CHARS = 400


def evaluate_jd_constraints(
    jd: ParsedJD, profile: CandidateProfile
) -> Tuple[bool, Optional[str]]:
    """
    Deterministic Gatekeeper sensor that enforces hard candidate boundaries
    before spending tokens on lead enrichment or outreach drafting.
    """
    rules = profile.qualification_rules
    title_lower = jd.role_title.lower()

    # 1. Check excluded role levels
    if jd.role_level in ("intern", "senior", "staff"):
        return False, f"Excluded role level detected: '{jd.role_level}'."

    for excluded_keyword in rules.excluded_role_levels:
        if excluded_keyword in title_lower.split():
            return False, f"Role title contains excluded keyword: '{excluded_keyword}'."

    # 2. Check strict B.Tech/B.E. gate against B.Sc degree
    if rules.reject_if_strict_btech_be_only and jd.strict_btech_be_only:
        return False, "JD strictly mandates B.Tech/B.E. and excludes B.Sc graduates."

    # 3. Check minimum required YOE ceiling
    if (
        jd.min_required_yoe is not None
        and jd.min_required_yoe > rules.max_allowed_required_yoe
    ):
        return (
            False,
            f"Required YOE ({jd.min_required_yoe} yrs) exceeds ceiling ({rules.max_allowed_required_yoe} yrs).",
        )

    return True, None


def validate_outreach_draft(
    draft: OutreachDraftPayload, profile: CandidateProfile
) -> HarnessCheckResult:
    """
    Deterministic Harness Sensor that inspects cold email and LinkedIn outreach drafts
    against the immutable CandidateProfile before the LLM Critic runs.
    """
    violations: list[str] = []
    allowed_skills = profile.flat_allowed_skills

    # 1. Skill Allowlist Check
    for skill in draft.claimed_skills:
        if skill.lower().strip() not in allowed_skills:
            violations.append(
                f"Unauthorized skill claimed in outreach: '{skill}' is not in candidate_profile.yaml."
            )

    # 2. Referenced Project ID Check
    if not draft.referenced_project_ids:
        violations.append("Outreach draft must reference at least 1 valid project_id.")
    for pid in draft.referenced_project_ids:
        if pid not in profile.valid_project_ids:
            violations.append(f"Hallucinated project_id referenced: '{pid}'.")

    # 3. URL Integrity Check
    for url in draft.included_urls:
        if url.rstrip("/") not in profile.valid_urls:
            violations.append(f"Unverified URL in outreach: '{url}'.")

    # 4. Length & Brevity Guardrails
    email_word_count = len(draft.email_body.split())
    if email_word_count > MAX_EMAIL_WORDS:
        violations.append(
            f"Cold email is too long ({email_word_count} words). Keep it at or below {MAX_EMAIL_WORDS} words."
        )

    if len(draft.linkedin_note) > MAX_LINKEDIN_CHARS:
        violations.append(
            f"LinkedIn note is too long ({len(draft.linkedin_note)} chars). Max allowed is {MAX_LINKEDIN_CHARS} chars."
        )

    return HarnessCheckResult(
        passed=(len(violations) == 0),
        violations=violations,
    )