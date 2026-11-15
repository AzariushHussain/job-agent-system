from typing import Tuple, Optional, List, Dict
from src.job_agent.graph.state import (
    CandidateProfile,
    ParsedJD,
    OutreachDraftPayload,
    HarnessCheckResult,
)

MAX_EMAIL_WORDS = 180
MAX_LINKEDIN_CHARS = 400

# Maps common JD variations to canonical lowercase skill names in candidate_profile.yaml
SKILL_ALIASES: Dict[str, str] = {
    "react": "react.js",
    "reactjs": "react.js",
    "react native": "react native",
    "node": "node.js",
    "nodejs": "node.js",
    "express": "express.js",
    "expressjs": "express.js",
    "postgres": "postgresql",
    "psql": "postgresql",
    "rest": "rest apis",
    "rest api": "rest apis",
    "restful api": "rest apis",
    "restful apis": "rest apis",
    "oop": "object-oriented programming",
    "lambda": "aws lambda",
    "sqs": "aws sqs",
    "ec2": "aws ec2",
    "s3": "aws s3",
    "openai": "openai apis",
    "llama2": "llama 2",
    "llama-2": "llama 2",
    "langchain": "langchain",
    "vector db": "vector dbs",
    "vector database": "vector dbs",
    "vector databases": "vector dbs",
    "websocket": "websockets",
    "socket.io": "socket.io",
    "socketio": "socket.io",
    "pubsub": "redis pub/sub",
    "pub/sub": "redis pub/sub",
    "js": "javascript",
    "html": "html5",
    "css": "css3",
}


def normalize_skill(skill: str) -> str:
    cleaned = skill.lower().strip()
    return SKILL_ALIASES.get(cleaned, cleaned)


def compute_skill_overlap(
    jd: ParsedJD, profile: CandidateProfile
) -> Tuple[float, List[str], List[str]]:
    """
    Computes the deterministic overlap between JD must-have skills and
    the candidate's allowed_skills allowlist.
    Returns: (match_ratio, matched_skills, missing_must_have_skills)
    """
    allowed = {normalize_skill(s) for s in profile.flat_allowed_skills}

    if not jd.must_have_skills:
        return 1.0, [], []

    matched: List[str] = []
    missing: List[str] = []

    for raw_skill in jd.must_have_skills:
        norm = normalize_skill(raw_skill)
        if norm in allowed:
            matched.append(raw_skill)
        else:
            missing.append(raw_skill)

    ratio = len(matched) / len(jd.must_have_skills)
    return round(ratio, 2), matched, missing


def rank_experiences_and_projects(
    jd: ParsedJD, profile: CandidateProfile
) -> Tuple[List[str], List[str]]:
    """
    Scores each work experience and project in candidate_profile.yaml against
    the JD's must_have_skills + nice_to_have_skills and returns the best matches.
    """
    jd_target_skills = {
        normalize_skill(s) for s in (jd.must_have_skills + jd.nice_to_have_skills)
    }

    # Score work experiences (keep top 2 with non-zero overlap, always include exp_lyfverse as primary FT role if tie)
    exp_scores: List[Tuple[int, str]] = []
    for exp in profile.work_experience:
        exp_skills = {normalize_skill(s) for s in exp.tech_stack}
        overlap_count = len(jd_target_skills & exp_skills)
        exp_scores.append((overlap_count, exp.exp_id))

    exp_scores.sort(key=lambda x: x[0], reverse=True)
    matched_exp_ids = [exp_id for score, exp_id in exp_scores if score > 0][:2]
    if not matched_exp_ids and profile.work_experience:
        matched_exp_ids = [profile.work_experience[0].exp_id]

    # Score projects (keep top 1-2)
    proj_scores: List[Tuple[int, str]] = []
    for proj in profile.projects:
        proj_skills = {normalize_skill(s) for s in proj.tech_stack}
        overlap_count = len(jd_target_skills & proj_skills)
        proj_scores.append((overlap_count, proj.project_id))

    proj_scores.sort(key=lambda x: x[0], reverse=True)
    matched_proj_ids = [pid for score, pid in proj_scores if score > 0][:2]
    if not matched_proj_ids and profile.projects:
        matched_proj_ids = [profile.projects[0].project_id]

    return matched_exp_ids, matched_proj_ids


def evaluate_jd_constraints(
    jd: ParsedJD, profile: CandidateProfile
) -> Tuple[bool, Optional[str], float, List[str], List[str]]:
    """
    Full Gatekeeper & Tech Stack Matching Sensor:
    Returns (is_qualified, reason, tech_match_score, matched_skills, missing_skills)
    """
    rules = profile.qualification_rules
    title_lower = jd.role_title.lower()

    # 1. Excluded Role Level Check
    if jd.role_level in ("intern", "senior", "staff"):
        return False, f"Excluded role level detected: '{jd.role_level}'.", 0.0, [], []

    for excluded_keyword in rules.excluded_role_levels:
        if excluded_keyword in title_lower.split():
            return (
                False,
                f"Role title contains excluded keyword: '{excluded_keyword}'.",
                0.0,
                [],
                [],
            )

    # 2. Role Category / Track Check
    if jd.role_category not in rules.target_role_types:
        return (
            False,
            f"Role category '{jd.role_category}' is outside target tracks ({', '.join(rules.target_role_types)}).",
            0.0,
            [],
            [],
        )

    # 3. Strict B.Tech/B.E. Check
    if rules.reject_if_strict_btech_be_only and jd.strict_btech_be_only:
        return (
            False,
            "JD strictly mandates B.Tech/B.E. and excludes B.Sc graduates.",
            0.0,
            [],
            [],
        )

    # 4. YOE Ceiling Check
    if (
        jd.min_required_yoe is not None
        and jd.min_required_yoe > rules.max_allowed_required_yoe
    ):
        return (
            False,
            f"Required YOE ({jd.min_required_yoe} yrs) exceeds ceiling ({rules.max_allowed_required_yoe} yrs).",
            0.0,
            [],
            [],
        )

    # 5. Tech Stack Overlap & Core Anchor Check
    match_ratio, matched_skills, missing_skills = compute_skill_overlap(jd, profile)

    all_jd_skills_norm = {
        normalize_skill(s) for s in (jd.must_have_skills + jd.nice_to_have_skills)
    }
    core_anchors_norm = {normalize_skill(s) for s in rules.core_anchor_skills}

    if all_jd_skills_norm and not (all_jd_skills_norm & core_anchors_norm):
        return (
            False,
            f"Zero overlap with core anchor stack ({', '.join(rules.core_anchor_skills)}). JD requires: {', '.join(jd.must_have_skills)}.",
            match_ratio,
            matched_skills,
            missing_skills,
        )

    if match_ratio < rules.min_must_have_match_ratio:
        return (
            False,
            f"Tech stack match score ({match_ratio:.0%}) is below minimum threshold ({rules.min_must_have_match_ratio:.0%}). Missing: {', '.join(missing_skills)}.",
            match_ratio,
            matched_skills,
            missing_skills,
        )

    return True, None, match_ratio, matched_skills, missing_skills


def validate_outreach_draft(
    draft: OutreachDraftPayload, profile: CandidateProfile
) -> HarnessCheckResult:
    """
    Deterministic Harness Sensor that inspects cold email and LinkedIn outreach drafts.
    """
    violations: list[str] = []
    allowed_skills = {normalize_skill(s) for s in profile.flat_allowed_skills}

    # 1. Skill Allowlist Check (Normalized)
    for skill in draft.claimed_skills:
        if normalize_skill(skill) not in allowed_skills:
            violations.append(
                f"Unauthorized skill claimed in outreach: '{skill}' is not in candidate_profile.yaml."
            )

    # 2. Referenced Experience & Project ID Check
    for exp_id in draft.referenced_experience_ids:
        if exp_id not in profile.valid_experience_ids:
            violations.append(f"Hallucinated exp_id referenced: '{exp_id}'.")

    if not draft.referenced_project_ids and not draft.referenced_experience_ids:
        violations.append(
            "Outreach draft must reference at least 1 valid exp_id or project_id."
        )

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
