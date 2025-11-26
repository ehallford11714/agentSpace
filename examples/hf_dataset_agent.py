"""Hugging Face dataset retrieval agent using a STELLA-like loop.

This example shows how a small set of cooperating agents can discover the
metadata for a public Hugging Face dataset, plan how to retrieve the desired
split, execute the download, and audit the result. The agents share a pivot
object that acts as the single source of truth for decisions and logs.

Usage:
    python examples/hf_dataset_agent.py           # uses ag_news train split sample
    python examples/hf_dataset_agent.py imdb test # different dataset and split

Dependencies:
    pip install datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, get_dataset_config_names, get_dataset_split_names, load_dataset
from pydantic import BaseModel, Field


# ==========================================
# 1. THE STELLA PIVOT (Shared State)
# ==========================================


class Issue(BaseModel):
    id: str
    issue_type: str
    severity: str
    description: str


class PlanStep(BaseModel):
    step_id: int
    target_issue_id: str
    action_type: str
    description: str
    rationale: str


class ActionLog(BaseModel):
    step_id: int
    code_executed: str
    success: bool
    error_message: Optional[str] = None
    rows_affected: Optional[int] = 0


@dataclass
class DatasetRequest:
    repo_id: str
    config: Optional[str] = None
    split: str = "train"
    sample_size: int = 500


class DatasetSnapshot(BaseModel):
    repo_id: str
    config: str
    split: str
    num_rows: int
    columns: List[str]


class DatasetPivot(BaseModel):
    """Single source of truth for the dataset retrieval agents."""

    pivot_id: str
    request: DatasetRequest
    available_configs: List[str] = Field(default_factory=list)
    available_splits: Dict[str, List[str]] = Field(default_factory=dict)

    dataset: Optional[DatasetSnapshot] = None
    known_issues: List[Issue] = Field(default_factory=list)
    current_plan: Optional[PlanStep] = None
    action_log: List[ActionLog] = Field(default_factory=list)
    status: str = "scanning"


# ==========================================
# 2. THE NETS (The Agents)
# ==========================================


class ScoutNet:
    """Observer: fetch dataset metadata and surface issues."""

    def run(self, pivot: DatasetPivot) -> DatasetPivot:
        print("🔎 [Scout] Inspecting Hugging Face repository metadata...")
        issues: List[Issue] = []

        try:
            config_names = get_dataset_config_names(pivot.request.repo_id)
        except Exception as exc:  # noqa: BLE001 - display hub errors to the user
            issues.append(
                Issue(
                    id="issue_repo_missing",
                    issue_type="repo_not_found",
                    severity="high",
                    description=f"Unable to resolve dataset repo: {exc}",
                )
            )
            pivot.known_issues = issues
            pivot.available_configs = []
            pivot.status = "planning"
            return pivot

        pivot.available_configs = config_names or [None]
        requested_config = pivot.request.config or (config_names[0] if config_names else None)

        if len(config_names) > 1 and pivot.request.config is None:
            issues.append(
                Issue(
                    id="issue_choose_config",
                    issue_type="config_selection",
                    severity="medium",
                    description="Multiple configs available; none selected.",
                )
            )

        if requested_config is None:
            issues.append(
                Issue(
                    id="issue_no_config",
                    issue_type="config_unavailable",
                    severity="high",
                    description="No configuration could be determined for the dataset.",
                )
            )
            pivot.known_issues = issues
            pivot.status = "planning"
            return pivot

        pivot.available_splits[requested_config] = get_dataset_split_names(pivot.request.repo_id, requested_config)

        if pivot.request.split not in pivot.available_splits[requested_config]:
            issues.append(
                Issue(
                    id="issue_missing_split",
                    issue_type="missing_split",
                    severity="high",
                    description=f"Requested split '{pivot.request.split}' is not available.",
                )
            )

        if pivot.dataset is None:
            issues.append(
                Issue(
                    id="issue_not_downloaded",
                    issue_type="not_retrieved",
                    severity="high",
                    description="Dataset has not been downloaded yet.",
                )
            )

        pivot.known_issues = issues
        pivot.status = "planning" if issues else "completed"
        return pivot


class StrategistNet:
    """Planner: determine the next best action to satisfy the request."""

    def run(self, pivot: DatasetPivot) -> DatasetPivot:
        if not pivot.known_issues:
            pivot.status = "completed"
            return pivot

        issue = pivot.known_issues[0]
        print(f"🧠 [Strategist] Planning fix for issue: {issue.issue_type}")

        if issue.issue_type == "config_selection":
            description = "Select the first available configuration as default."
            action = "set_config"
        elif issue.issue_type == "missing_split":
            description = "Switch to the first available split for the chosen config."
            action = "set_split"
        elif issue.issue_type == "not_retrieved":
            description = "Download the requested dataset split and trim to sample size."
            action = "download_dataset"
        elif issue.issue_type == "repo_not_found":
            description = "Stop because the dataset repository is unavailable."
            action = "abort"
        else:
            description = "Abort due to unresolved configuration issues."
            action = "abort"

        pivot.current_plan = PlanStep(
            step_id=len(pivot.action_log) + 1,
            target_issue_id=issue.id,
            action_type=action,
            description=description,
            rationale="Follow STELLA loop to satisfy dataset request safely.",
        )
        pivot.status = "executing"
        return pivot


class OperatorNet:
    """Executor: perform the planned action."""

    def run(self, pivot: DatasetPivot) -> Tuple[Optional[Dataset], DatasetPivot]:
        plan = pivot.current_plan
        if plan is None:
            raise ValueError("No plan to execute")

        dataset: Optional[Dataset] = None
        print(f"🔧 [Operator] Action: {plan.description}")

        try:
            if plan.action_type == "set_config":
                pivot.request.config = pivot.available_configs[0]
                code = "request.config = available_configs[0]"
                pivot.status = "scanning"
            elif plan.action_type == "set_split":
                pivot.request.split = pivot.available_splits[pivot.request.config or pivot.available_configs[0]][0]
                code = "request.split = first_available_split"
                pivot.status = "scanning"
            elif plan.action_type == "download_dataset":
                dataset = load_dataset(
                    pivot.request.repo_id,
                    pivot.request.config,
                    split=pivot.request.split,
                )
                if pivot.request.sample_size and len(dataset) > pivot.request.sample_size:
                    dataset = dataset.select(range(pivot.request.sample_size))

                pivot.dataset = DatasetSnapshot(
                    repo_id=pivot.request.repo_id,
                    config=pivot.request.config or pivot.available_configs[0],
                    split=pivot.request.split,
                    num_rows=len(dataset),
                    columns=list(dataset.features.keys()),
                )
                code = "dataset = load_dataset(...); optional select(sample_size)"
                pivot.status = "auditing"
            else:
                code = "abort"
                pivot.status = "failed"
                raise RuntimeError("Aborting per plan")

            pivot.known_issues = [i for i in pivot.known_issues if i.id != plan.target_issue_id]
            pivot.current_plan = None

            pivot.action_log.append(
                ActionLog(
                    step_id=plan.step_id,
                    code_executed=code,
                    success=True,
                    rows_affected=pivot.dataset.num_rows if pivot.dataset else 0,
                )
            )
        except Exception as exc:  # noqa: BLE001 - operator should expose actual error
            pivot.action_log.append(
                ActionLog(
                    step_id=plan.step_id,
                    code_executed=plan.action_type,
                    success=False,
                    error_message=str(exc),
                )
            )
            pivot.status = "failed"
            print(f"❌ [Operator] Error executing plan: {exc}")

        return dataset, pivot


class AuditorNet:
    """Critic: check post-conditions and mark completion."""

    def run(self, pivot: DatasetPivot) -> DatasetPivot:
        print("⚖️ [Auditor] Verifying retrieved dataset...")

        if pivot.dataset is None:
            pivot.status = "failed"
            print("❌ [Auditor] No dataset snapshot found; retrieval failed.")
            return pivot

        if pivot.dataset.num_rows == 0:
            pivot.status = "failed"
            print("❌ [Auditor] Dataset is empty after retrieval.")
            return pivot

        print(
            f"✅ [Auditor] Retrieved {pivot.dataset.num_rows} rows "
            f"with columns: {', '.join(pivot.dataset.columns)}"
        )
        pivot.status = "completed"
        return pivot


# ==========================================
# 3. ORCHESTRATION LOOP
# ==========================================


def run_stella_loop(request: DatasetRequest) -> Tuple[Optional[Dataset], DatasetPivot]:
    pivot = DatasetPivot(
        pivot_id="hf_job_001",
        request=request,
    )

    scout = ScoutNet()
    strategist = StrategistNet()
    operator = OperatorNet()
    auditor = AuditorNet()

    dataset: Optional[Dataset] = None
    steps = 0
    max_steps = 12

    print(f"🚀 HF Agent initialized for repo '{request.repo_id}'.\n")

    while pivot.status not in {"completed", "failed"} and steps < max_steps:
        steps += 1
        print(f"\n--- Cycle {steps} [State: {pivot.status}] ---")

        if pivot.status == "scanning":
            pivot = scout.run(pivot)
        elif pivot.status == "planning":
            pivot = strategist.run(pivot)
        elif pivot.status == "executing":
            dataset, pivot = operator.run(pivot)
        elif pivot.status == "auditing":
            pivot = auditor.run(pivot)

    if pivot.status == "completed":
        print("\n✨ Dataset retrieved successfully!")
    else:
        print("\n💀 Retrieval process failed or timed out.")

    return dataset, pivot


# ==========================================
# 4. DEMO ENTRYPOINT
# ==========================================


def main(repo_id: str = "ag_news", config: Optional[str] = None, split: str = "train", sample_size: int = 200):
    request = DatasetRequest(repo_id=repo_id, config=config, split=split, sample_size=sample_size)
    dataset, pivot = run_stella_loop(request)

    if dataset is not None:
        print("\n--- Sample rows ---")
        print(dataset.to_pandas().head())

    print("\n--- Action Log ---")
    for log in pivot.action_log:
        print(f"Step {log.step_id}: {log.code_executed} -> Success: {log.success}")


if __name__ == "__main__":
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else "ag_news"
    cfg = sys.argv[2] if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    main(repo, cfg, split)
