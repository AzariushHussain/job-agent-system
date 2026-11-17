from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from src.job_agent.api.main import app
from src.job_agent.graph.workflow import compile_job_application_graph
from src.job_agent.graph.state import (
    CriticEvaluation,
    OutreachDraftPayload,
    ParsedJD,
)


class _MockLLM:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, _messages):
        return self._payload


def test_fastapi_evaluate_and_hitl_decision_endpoints(tmp_path):
    app.state.graph = compile_job_application_graph(
        gatekeeper_llm=_MockLLM(
            ParsedJD(
                company_name="ApiTestCorp",
                role_title="Backend Engineer",
                location="Bangalore",
                role_category="backend",
                min_required_yoe=1.0,
                strict_btech_be_only=False,
                role_level="entry",
                must_have_skills=["Python", "FastAPI", "Redis"],
            )
        ),
        writer_llm=_MockLLM(
            OutreachDraftPayload(
                recipient_role_or_name="CTO",
                recipient_email="cto@apitestcorp.io",
                email_subject="Backend Engineer — Python, FastAPI & Redis",
                email_body="Hi, built FastAPI and Redis systems at Lyfverse saving $2,900+/yr.",
                linkedin_note="Hi! Applying for the Backend Engineer role.",
                claimed_skills=["Python", "FastAPI", "Redis"],
                referenced_experience_ids=["exp_lyfverse"],
                referenced_project_ids=["proj_quickchat"],
                included_urls=["https://github.com/AzariushHussain"],
            )
        ),
        critic_llm=_MockLLM(
            CriticEvaluation(passed=True, grounding_score=0.98, specificity_score=0.92)
        ),
        mx_resolver=lambda d: ["mx.apitestcorp.io"],
        smtp_prober=lambda email, mx: "smtp_verified",
        output_dir=tmp_path,
        checkpointer=MemorySaver(),
        enable_hitl=True,
    )

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    eval_resp = client.post(
        "/jobs/evaluate",
        json={
            "job_url": "https://apitestcorp.io/jobs/1",
            "raw_jd_text": "Hiring Backend Engineer in Bangalore with Python, FastAPI, Redis. Email cto@apitestcorp.io",
            "thread_id": "api_test_thread_1",
        },
    )
    assert eval_resp.status_code == 200
    data = eval_resp.json()
    assert data["is_qualified"] is True
    assert data["waiting_for_human_approval"] is True
    assert data["discovered_emails"] == ["cto@apitestcorp.io"]

    dec_resp = client.post(
        "/jobs/api_test_thread_1/decision",
        json={"decision": "approve"},
    )
    assert dec_resp.status_code == 200
    assert dec_resp.json()["dispatch_status"] == "EXPORTED"
    assert (tmp_path / "vetted_job_leads.xlsx").exists()
