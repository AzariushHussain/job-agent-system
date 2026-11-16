from pathlib import Path
from typing import Literal, Optional, Any, Callable, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.job_agent.core.config import settings
from src.job_agent.graph.state import CandidateProfile, JobApplicationState
from src.job_agent.graph.nodes.scout import build_scout_node
from src.job_agent.graph.nodes.gatekeeper import build_gatekeeper_node
from src.job_agent.graph.nodes.outreach_writer import build_outreach_writer_node
from src.job_agent.graph.nodes.critic import build_critic_node
from src.job_agent.graph.nodes.dispatch import build_dispatch_node


def route_after_scout(
    state: JobApplicationState,
) -> Literal["gatekeeper", "end"]:
    if state.get("scrape_error"):
        return "end"
    return "gatekeeper"


def route_after_gatekeeper(
    state: JobApplicationState,
) -> Literal["outreach_writer", "end"]:
    if state.get("is_qualified"):
        return "outreach_writer"
    return "end"


def route_after_critic(
    state: JobApplicationState,
) -> Literal["outreach_writer", "dispatch_lead"]:
    critic_eval = state.get("critic_eval")
    revisions = state.get("revision_count", 0)
    if (critic_eval and critic_eval.passed) or revisions >= settings.max_revisions:
        return "dispatch_lead"
    return "outreach_writer"


def compile_job_application_graph(
    profile: Optional[CandidateProfile] = None,
    gatekeeper_llm: Optional[Any] = None,
    writer_llm: Optional[Any] = None,
    critic_llm: Optional[Any] = None,
    http_fetcher: Optional[Callable[[str], str]] = None,
    mx_resolver: Optional[Callable[[str], List[str]]] = None,
    smtp_prober: Optional[Callable[[str, str], str]] = None,
    output_dir: str | Path = "outputs",
    checkpointer: Optional[BaseCheckpointSaver] = None,
    enable_hitl: bool = False,
):
    loaded_profile = profile or CandidateProfile.from_yaml(settings.profile_path)

    builder = StateGraph(JobApplicationState)

    builder.add_node(
        "scout",
        build_scout_node(
            http_fetcher=http_fetcher,
            mx_resolver=mx_resolver,
            smtp_prober=smtp_prober,
        ),
    )
    builder.add_node(
        "gatekeeper", build_gatekeeper_node(loaded_profile, gatekeeper_llm)
    )
    builder.add_node(
        "outreach_writer", build_outreach_writer_node(loaded_profile, writer_llm)
    )
    builder.add_node("critic", build_critic_node(loaded_profile, critic_llm))
    builder.add_node(
        "dispatch_lead",
        build_dispatch_node(loaded_profile, output_dir=output_dir),
    )

    builder.add_edge(START, "scout")
    builder.add_conditional_edges(
        "scout",
        route_after_scout,
        {"gatekeeper": "gatekeeper", "end": END},
    )
    builder.add_conditional_edges(
        "gatekeeper",
        route_after_gatekeeper,
        {"outreach_writer": "outreach_writer", "end": END},
    )
    builder.add_edge("outreach_writer", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "outreach_writer": "outreach_writer",
            "dispatch_lead": "dispatch_lead",
        },
    )
    builder.add_edge("dispatch_lead", END)

    interrupt_nodes = ["dispatch_lead"] if enable_hitl else None
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes,
    )
