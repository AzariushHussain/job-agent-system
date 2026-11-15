from typing import Literal, Optional, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.job_agent.core.config import settings
from src.job_agent.graph.state import CandidateProfile, JobApplicationState
from src.job_agent.graph.nodes.gatekeeper import build_gatekeeper_node
from src.job_agent.graph.nodes.outreach_writer import build_outreach_writer_node
from src.job_agent.graph.nodes.critic import build_critic_node


def route_after_gatekeeper(
    state: JobApplicationState,
) -> Literal["outreach_writer", "end"]:
    if state.get("is_qualified"):
        return "outreach_writer"
    return "end"


def route_after_critic(
    state: JobApplicationState,
) -> Literal["outreach_writer", "end"]:
    critic_eval = state.get("critic_eval")
    revisions = state.get("revision_count", 0)
    if (critic_eval and critic_eval.passed) or revisions >= settings.max_revisions:
        return "end"
    return "outreach_writer"


def compile_job_application_graph(
    profile: Optional[CandidateProfile] = None,
    gatekeeper_llm: Optional[Any] = None,
    writer_llm: Optional[Any] = None,
    critic_llm: Optional[Any] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    loaded_profile = profile or CandidateProfile.from_yaml(settings.profile_path)

    builder = StateGraph(JobApplicationState)

    builder.add_node(
        "gatekeeper", build_gatekeeper_node(loaded_profile, gatekeeper_llm)
    )
    builder.add_node(
        "outreach_writer", build_outreach_writer_node(loaded_profile, writer_llm)
    )
    builder.add_node(
        "critic", build_critic_node(loaded_profile, critic_llm)
    )

    builder.add_edge(START, "gatekeeper")
    builder.add_conditional_edges(
        "gatekeeper",
        route_after_gatekeeper,
        {"outreach_writer": "outreach_writer", "end": END},
    )
    builder.add_edge("outreach_writer", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"outreach_writer": "outreach_writer", "end": END},
    )

    return builder.compile(checkpointer=checkpointer)
