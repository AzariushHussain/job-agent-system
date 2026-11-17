from pathlib import Path
import yaml
import pytest
from src.job_agent.core.config import settings
from src.job_agent.graph.state import CandidateProfile, ParsedJD
from src.job_agent.harness.validators import (
    evaluate_jd_constraints,
    rank_experiences_and_projects,
)

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_jds.yaml"


@pytest.fixture(scope="module")
def profile() -> CandidateProfile:
    return CandidateProfile.from_yaml(settings.profile_path)


@pytest.fixture(scope="module")
def golden_cases() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["cases"]


def test_golden_evaluation_pipeline_accuracy(
    profile: CandidateProfile, golden_cases: list[dict]
):
    correct_qualifications = 0
    ranking_checks = 0
    correct_rankings = 0

    for case in golden_cases:
        jd = ParsedJD(
            company_name=case["company_name"],
            role_title=case["role_title"],
            location=case["location"],
            role_category=case["role_category"],
            role_level=case["role_level"],
            min_required_yoe=case["min_required_yoe"],
            strict_btech_be_only=case["strict_btech_be_only"],
            must_have_skills=case["must_have_skills"],
            nice_to_have_skills=case.get("nice_to_have_skills", []),
        )

        is_qualified, reason, score, _, _ = evaluate_jd_constraints(jd, profile)
        if is_qualified == case["expected_qualified"]:
            correct_qualifications += 1
        else:
            pytest.fail(
                f"[{case['case_id']}] Qualification mismatch: expected {case['expected_qualified']}, "
                f"got {is_qualified} (reason={reason}, score={score})"
            )

        matched_exps, matched_projs = rank_experiences_and_projects(jd, profile)

        if "expected_top_experience" in case:
            ranking_checks += 1
            if matched_exps and matched_exps[0] == case["expected_top_experience"]:
                correct_rankings += 1
            else:
                pytest.fail(
                    f"[{case['case_id']}] Top experience mismatch: expected {case['expected_top_experience']}, got {matched_exps}"
                )

        if "expected_top_project" in case:
            ranking_checks += 1
            if matched_projs and matched_projs[0] == case["expected_top_project"]:
                correct_rankings += 1
            else:
                pytest.fail(
                    f"[{case['case_id']}] Top project mismatch: expected {case['expected_top_project']}, got {matched_projs}"
                )

    qual_accuracy = correct_qualifications / len(golden_cases)
    rank_accuracy = correct_rankings / ranking_checks if ranking_checks else 1.0

    assert qual_accuracy == 1.0
    assert rank_accuracy == 1.0
