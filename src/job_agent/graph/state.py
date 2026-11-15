from typing import TypedDict, Annotated, List, Dict, Optional, Literal
from operator import add
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


# =====================================================================
# 1. CANDIDATE PROFILE MODELS (Loaded from candidate_profile.yaml)
# =====================================================================
class QualificationRules(BaseModel):
    total_yoe_years: float
    max_allowed_required_yoe: float
    target_role_types: List[str]
    excluded_role_levels: List[str]
    reject_if_strict_btech_be_only: bool


class WorkExperienceItem(BaseModel):
    exp_id: str
    company: str
    role: str
    start_date: str
    end_date: str
    locked_metrics: List[str] = Field(default_factory=list)
    base_bullets: List[str]


class ProjectItem(BaseModel):
    project_id: str
    name: str
    repo_url: str
    tech_stack: List[str]
    locked_metrics: List[str] = Field(default_factory=list)
    base_bullets: List[str]


class CandidateProfile(BaseModel):
    candidate: Dict[str, object]
    education: Dict[str, object]
    qualification_rules: QualificationRules
    allowed_skills: Dict[str, List[str]]
    work_experience: List[WorkExperienceItem]
    projects: List[ProjectItem]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CandidateProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @property
    def flat_allowed_skills(self) -> set[str]:
        skills = set()
        for category_list in self.allowed_skills.values():
            for skill in category_list:
                skills.add(skill.lower().strip())
        return skills

    @property
    def valid_project_ids(self) -> set[str]:
        return {p.project_id for p in self.projects}

    @property
    def valid_urls(self) -> set[str]:
        urls = {p.repo_url.rstrip("/") for p in self.projects}
        for key in ("github_url", "portfolio_url"):
            if key in self.candidate:
                urls.add(str(self.candidate[key]).rstrip("/"))
        return urls


# =====================================================================
# 2. AGENT ARTIFACT SCHEMAS (Gatekeeper & Outreach Writer)
# =====================================================================
class ParsedJD(BaseModel):
    company_name: str
    role_title: str
    location: str
    is_remote: bool = False
    min_required_yoe: Optional[float] = Field(
        default=None, description="Minimum years of experience explicitly required"
    )
    strict_btech_be_only: bool = Field(
        default=False,
        description="True ONLY if the JD explicitly mandates B.Tech/B.E. and excludes B.Sc/BCA",
    )
    role_level: Literal["intern", "entry", "mid", "senior", "staff", "unknown"] = "entry"
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    matched_project_ids: List[str] = Field(
        default_factory=list,
        description="1 to 2 project_ids from candidate_profile most relevant to this JD",
    )


class OutreachDraftPayload(BaseModel):
    recipient_role_or_name: str = Field(
        default="Hiring Team", description="Target recipient (e.g., CTO, Engineering Lead, Hiring Team)"
    )
    recipient_email: Optional[str] = Field(
        default=None, description="Discovered direct or career/support email if available"
    )
    email_subject: str
    email_body: str = Field(
        description="Concise cold email body (under 180 words) referencing matched projects and attached resume"
    )
    linkedin_note: str = Field(
        description="Short LinkedIn connection/DM pitch (under 400 characters)"
    )
    claimed_skills: List[str] = Field(
        description="Skills explicitly claimed in the outreach message; must exist in allowed_skills"
    )
    referenced_project_ids: List[str] = Field(
        description="project_ids cited in the outreach draft"
    )
    included_urls: List[str] = Field(
        default_factory=list,
        description="All portfolio or GitHub URLs included in the outreach text",
    )


class HarnessCheckResult(BaseModel):
    passed: bool
    violations: List[str] = Field(default_factory=list)


class CriticEvaluation(BaseModel):
    passed: bool
    grounding_score: float = Field(ge=0.0, le=1.0)
    specificity_score: float = Field(ge=0.0, le=1.0)
    revision_instructions: Optional[str] = None


# =====================================================================
# 3. LANGGRAPH WORKING MEMORY STATE
# =====================================================================
class JobApplicationState(TypedDict):
    job_url: str
    raw_jd_text: str

    # Gatekeeper outputs
    parsed_jd: Optional[ParsedJD]
    is_qualified: Optional[bool]
    disqualification_reason: Optional[str]

    # Outreach Writer & Critic loop outputs
    outreach_draft: Optional[OutreachDraftPayload]
    harness_report: Optional[HarnessCheckResult]
    critic_eval: Optional[CriticEvaluation]
    revision_count: int

    # Human-in-the-loop & export/dispatch
    human_decision: Optional[Literal["approve", "edit", "skip"]]
    dispatch_status: Optional[str]

    # Append-only audit log
    audit_log: Annotated[List[str], add]