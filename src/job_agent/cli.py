import argparse
import json
from pathlib import Path
from src.job_agent.graph.workflow import compile_job_application_graph
from src.job_agent.graph.state import (
    CriticEvaluation,
    OutreachDraftPayload,
    ParsedJD,
)


class _DemoLLM:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, _messages):
        return self._payload


def main():
    parser = argparse.ArgumentParser(
        description="Run the LangGraph Job Application Pipeline and export vetted leads to Excel/CSV."
    )
    parser.add_argument("--url", type=str, default="", help="Job posting URL")
    parser.add_argument("--jd-file", type=str, default="", help="Path to raw JD text file")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a local offline demo and generate outputs/vetted_job_leads.xlsx",
    )
    args = parser.parse_args()

    if args.demo:
        graph = compile_job_application_graph(
            gatekeeper_llm=_DemoLLM(
                ParsedJD(
                    company_name="Nexusolve AI",
                    role_title="Full-Stack & AI Engineer",
                    location="Bangalore",
                    role_category="full-stack",
                    min_required_yoe=1.5,
                    strict_btech_be_only=False,
                    role_level="entry",
                    must_have_skills=["Python", "FastAPI", "Django", "React.js", "PostgreSQL", "Redis", "RAG"],
                )
            ),
            writer_llm=_DemoLLM(
                OutreachDraftPayload(
                    recipient_role_or_name="Engineering Lead",
                    recipient_email="careers@nexusolve.io",
                    email_subject="Full-Stack & AI Engineer — Python, FastAPI, Django, React.js & RAG",
                    email_body=(
                        "Hi Team, I'm applying for the Full-Stack & AI Engineer role at Nexusolve AI. "
                        "Across ~2 years at Lyfverse Technologies, Friggly.ai, and Vibtree, I've built "
                        "FastAPI/Django backends and React.js apps, integrated 4 payment gateways, "
                        "cut Cloudflare image costs by 100% ($2,900+/yr saved), and shipped RAG pipelines "
                        "with LangChain and LLaMA 2. Attached is my resume!"
                    ),
                    linkedin_note=(
                        "Hi! Saw the Full-Stack & AI Engineer opening. I have ~2 yrs experience in Python, "
                        "FastAPI, Django, React.js & RAG (cut cloud image costs 100%). Sent my resume via email!"
                    ),
                    claimed_skills=["Python", "FastAPI", "Django", "React.js", "PostgreSQL", "Redis", "RAG", "LangChain", "LLaMA 2"],
                    referenced_experience_ids=["exp_lyfverse", "exp_vibtree"],
                    referenced_project_ids=["proj_quickchat"],
                    included_urls=["https://github.com/AzariushHussain"],
                )
            ),
            critic_llm=_DemoLLM(
                CriticEvaluation(
                    passed=True,
                    grounding_score=0.99,
                    specificity_score=0.95,
                    revision_instructions=None,
                )
            ),
            mx_resolver=lambda d: ["mx.nexusolve.io"],
            smtp_prober=lambda email, mx: "smtp_verified",
            enable_hitl=False,
        )
        raw_text = (
            "Hiring Full-Stack & AI Engineer in Bangalore (1-2 yrs). Stack: Python, FastAPI, Django, "
            "React.js, PostgreSQL, Redis, RAG. Contact careers@nexusolve.io."
        )
        job_url = "https://nexusolve.io/careers/fullstack-ai"
    else:
        raw_text = Path(args.jd_file).read_text(encoding="utf-8") if args.jd_file else ""
        job_url = args.url
        graph = compile_job_application_graph(enable_hitl=False)

    final_state = graph.invoke(
        {
            "job_url": job_url,
            "raw_jd_text": raw_text,
            "revision_count": 0,
            "audit_log": [],
        }
    )

    print(
        json.dumps(
            {
                "is_qualified": final_state.get("is_qualified"),
                "tech_match_score": final_state.get("tech_match_score"),
                "discovered_emails": final_state.get("discovered_emails"),
                "dispatch_status": final_state.get("dispatch_status"),
                "audit_log": final_state.get("audit_log"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
