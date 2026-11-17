import uuid
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver

from src.job_agent.graph.workflow import compile_job_application_graph

app = FastAPI(
    title="Job Application Agentic Workflow API",
    version="1.0.0",
)

memory_checkpointer = MemorySaver()
app.state.graph = compile_job_application_graph(
    checkpointer=memory_checkpointer,
    enable_hitl=True,
)


class JobRunRequest(BaseModel):
    job_url: str = ""
    raw_jd_text: str = ""
    thread_id: Optional[str] = None


class HumanDecisionRequest(BaseModel):
    decision: Literal["approve", "skip"] = "approve"


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/jobs/evaluate")
def evaluate_job(req: JobRunRequest):
    thread_id = req.thread_id or f"job_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    state = app.state.graph.invoke(
        {
            "job_url": req.job_url,
            "raw_jd_text": req.raw_jd_text,
            "revision_count": 0,
            "audit_log": [],
        },
        config=config,
    )

    snapshot = app.state.graph.get_state(config)
    waiting_for_human = "dispatch_lead" in snapshot.next

    return {
        "thread_id": thread_id,
        "is_qualified": state.get("is_qualified"),
        "disqualification_reason": state.get("disqualification_reason"),
        "tech_match_score": state.get("tech_match_score"),
        "matched_skills": state.get("matched_skills"),
        "discovered_emails": state.get("discovered_emails"),
        "waiting_for_human_approval": waiting_for_human,
        "outreach_draft": state.get("outreach_draft"),
        "audit_log": state.get("audit_log", []),
    }


@app.post("/jobs/{thread_id}/decision")
def submit_human_decision(thread_id: str, req: HumanDecisionRequest):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app.state.graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    app.state.graph.update_state(config, {"human_decision": req.decision})
    final_state = app.state.graph.invoke(None, config=config)

    return {
        "thread_id": thread_id,
        "human_decision": req.decision,
        "dispatch_status": final_state.get("dispatch_status"),
        "audit_log": final_state.get("audit_log", []),
    }
