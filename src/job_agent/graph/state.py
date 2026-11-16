from typing import TypedDict, Annotated, List, Dict, Optional, Literal
from operator import add
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class QualificationRules(BaseModel):
    total_yoe_years: float
    max_allowed_required_yoe: float
    min_must_have_match_ratio: float = 0.40
    core_anchor_skills: List[str] = Field(default_factory=list)
    target_role_types: List[str]
    excluded_role_levels: List[str]
    reject_if_strict_btech_be_only: bool


class WorkExperienceItem(BaseModel):
    exp_id: str
    company: str
    role: str
    start_date: str
    end_date: str
    tech_stack: List[str] = Field(default_factory=list)
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
    def valid_experience_ids(self) -> set[str]:
        return {e.exp_id for e in self.work_experience}

    @property
    def valid_project_ids(self) -> set[str]:
        return {p.project_id for p in self.projects}

    @property
    def valid_urls(self) -> set[str]:
        urls = {p.repo_url.rstrip("/") for p in self.projects}
        for key in ("github_url", "linkedin_url", "portfolio_url"):
            if key in self.candidate and self.candidate[key]:
                urls.add(str(self.candidate[key]).rstrip("/"))
        return urls


class ParsedJD(BaseModel):
    company_name: str
    role_title: str
    location: str
    is_remote: bool = False
    role_category: Literal[
        "backend",
        "full-stack",
        "software_engineer",
        "ai_engineer",
        "genai",
        "frontend_only",
        "mobile_only",
        "devops_sre",
        "qa_testing",
        "data_analyst",
        "other",
    ] = "software_engineer"
    min_required_yoe: Optional[float] = Field(default=None)
    strict_btech_be_only: bool = Field(default=False)
    role_level: Literal["intern", "entry", "mid", "senior", "staff", "unknown"] = "entry"
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    matched_experience_ids: List[str] = Field(default_factory=list)
    matched_project_ids: List[str] = Field(default_factory=list)


class OutreachDraftPayload(BaseModel):
    recipient_role_or_name: str = Field(default="Hiring Team")
    recipient_email: Optional[str] = Field(default=None)
    email_subject: str
    email_body: str
    linkedin_note: str
    claimed_skills: List[str]
    referenced_experience_ids: List[str] = Field(default_factory=list)
    referenced_project_ids: List[str] = Field(default_factory=list)
    included_urls: List[str] = Field(default_factory=list)


class HarnessCheckResult(BaseModel):
    passed: bool
    violations: List[str] = Field(default_factory=list)


class CriticEvaluation(BaseModel):
    passed: bool
    grounding_score: float = Field(ge=0.0, le=1.0)
    specificity_score: float = Field(ge=0.0, le=1.0)
    revision_instructions: Optional[str] = None


class JobApplicationState(TypedDict):
    job_url: str
    raw_jd_text: str

    discovered_emails: Optional[List[str]]
    scrape_error: Optional[str]

    parsed_jd: Optional[ParsedJD]
    is_qualified: Optional[bool]
    disqualification_reason: Optional[str]
    tech_match_score: Optional[float]
    matched_skills: Optional[List[str]]
    missing_must_have_skills: Optional[List[str]]

    outreach_draft: Optional[OutreachDraftPayload]
    harness_report: Optional[HarnessCheckResult]
    critic_eval: Optional[CriticEvaluation]
    revision_count: int

    human_decision: Optional[Literal["approve", "edit", "skip"]]
    dispatch_status: Optional[str]

    audit_log: Annotated[List[str], add]
