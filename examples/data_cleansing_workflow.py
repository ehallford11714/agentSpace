"""Data cleansing workflow example using simple cooperative agents.

This module demonstrates a small stateful pipeline that scans a pandas
``DataFrame`` for known quality issues, plans a fix, executes the plan,
and audits the results. It is intentionally written as a single,
self-contained example so it can be run directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel


# ==========================================
# 1. THE STELLA PIVOT (Shared State)
# ==========================================

class ColumnEntity(BaseModel):
    name: str
    dtype: str
    null_count: int
    sample_values: List[Any]


class DatasetEntity(BaseModel):
    name: str
    row_count: int
    columns: Dict[str, ColumnEntity]


class Issue(BaseModel):
    id: str
    column: Optional[str] = None
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


class DataCleansingPivot(BaseModel):
    """The single source of truth that all agents read/write."""

    pivot_id: str
    goal: str = "Prepare data for ML training"

    # ENTITIES: The facts about the data
    dataset_meta: DatasetEntity

    # CONSTRAINTS: The rules of the road
    constraints: List[str] = [
        "Do not drop more than 5% of total rows",
        "Preserve customer_id uniqueness",
    ]

    # STATE: What is currently wrong?
    known_issues: List[Issue] = []
    current_plan: Optional[PlanStep] = None

    # MEMORY: What have we done?
    action_log: List[ActionLog] = []
    status: str = "scanning"


# ==========================================
# 2. THE MOCK LLM INTERFACE (Replace this!)
# ==========================================


def call_llm(system_prompt: str, user_prompt: str, mock_response: str | None = None):
    """In production, replace this with an LLM call.

    For this demo, it returns ``mock_response`` to simulate intelligence.
    """

    return mock_response


# ==========================================
# 3. THE NETS (The Agents)
# ==========================================


class ScannerNet:
    """The Observer: updates dataset metadata and finds issues."""

    def run(self, df: pd.DataFrame, pivot: DataCleansingPivot) -> DataCleansingPivot:
        print("🔎 [Scanner] Profiling dataframe...")

        # 1. Update Entities (Metadata)
        pivot.dataset_meta.row_count = len(df)
        pivot.dataset_meta.columns = {}

        for col in df.columns:
            pivot.dataset_meta.columns[col] = ColumnEntity(
                name=col,
                dtype=str(df[col].dtype),
                null_count=int(df[col].isnull().sum()),
                sample_values=df[col].dropna().head(3).tolist(),
            )

        # 2. Heuristic Check (Simulating LLM Insight)
        pivot.known_issues = []

        if "total_charges" in df.columns and df["total_charges"].dtype == "O":
            pivot.known_issues.append(
                Issue(
                    id="iss_1",
                    column="total_charges",
                    issue_type="type_mismatch",
                    severity="high",
                    description="Column is Object but should be Numeric. Contains empty strings.",
                )
            )

        if "tenure" in df.columns and df["tenure"].isnull().sum() > 0:
            pivot.known_issues.append(
                Issue(
                    id="iss_2",
                    column="tenure",
                    issue_type="missing_values",
                    severity="medium",
                    description="Missing values found.",
                )
            )

        pivot.status = "planning" if pivot.known_issues else "completed"
        return pivot


class StrategistNet:
    """The Planner: decides what to do next based on the pivot."""

    def run(self, pivot: DataCleansingPivot) -> DataCleansingPivot:
        if not pivot.known_issues:
            pivot.status = "completed"
            return pivot

        print("🧠 [Strategist] Planning fix for top issue...")

        # Sort by severity (High > Medium > Low) — for the demo we take the first.
        target_issue = pivot.known_issues[0]

        system_prompt = (
            "You are a Data Strategy Agent. Given a list of data issues and "
            "constraints, propose a safe Python pandas operation to fix the highest "
            "priority issue."
        )
        pivot_json = pivot.model_dump_json(exclude={"dataset_meta"})

        _ = call_llm(system_prompt, pivot_json)  # Placeholder for future LLM wiring.

        if target_issue.issue_type == "type_mismatch":
            mock_plan_desc = "Convert to numeric using coerce errors, then fill NaNs with median."
        elif target_issue.issue_type == "missing_values":
            mock_plan_desc = "Impute missing values with the median of the column."
        else:
            mock_plan_desc = "Drop rows."

        pivot.current_plan = PlanStep(
            step_id=len(pivot.action_log) + 1,
            target_issue_id=target_issue.id,
            action_type="code_execution",
            description=mock_plan_desc,
            rationale="Constraints allow imputation; type conversion is required for ML.",
        )

        pivot.status = "executing"
        return pivot


class SurgeonNet:
    """The Executor: generates and runs code."""

    def run(self, df: pd.DataFrame, pivot: DataCleansingPivot) -> Tuple[pd.DataFrame, DataCleansingPivot]:
        plan = pivot.current_plan
        if plan is None:
            raise ValueError("No plan to execute")

        print(f"🔪 [Surgeon] Executing Plan: {plan.description}")

        try:
            rows_before = len(df)

            if "Convert to numeric" in plan.description:
                code = (
                    "df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce'); "
                    "df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())"
                )
                df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
                df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())

            elif "Impute missing" in plan.description:
                code = "df['tenure'] = df['tenure'].fillna(df['tenure'].median())"
                df["tenure"] = df["tenure"].fillna(df["tenure"].median())

            else:
                code = "# Unknown plan, pass"

            rows_after = len(df)

            pivot.action_log.append(
                ActionLog(
                    step_id=plan.step_id,
                    code_executed=code,
                    success=True,
                    rows_affected=rows_before - rows_after,
                )
            )

            pivot.known_issues = [i for i in pivot.known_issues if i.id != plan.target_issue_id]
            pivot.current_plan = None
            pivot.status = "auditing"

        except Exception as exc:  # pragma: no cover - demo logging only
            print(f"❌ [Surgeon] Error: {exc}")
            pivot.action_log.append(
                ActionLog(
                    step_id=plan.step_id,
                    code_executed="ERROR",
                    success=False,
                    error_message=str(exc),
                )
            )
            pivot.status = "failed"

        return df, pivot


class AuditorNet:
    """The Critic: checks safety and constraints."""

    def run(self, df: pd.DataFrame, pivot: DataCleansingPivot) -> DataCleansingPivot:
        print("⚖️ [Auditor] Verifying constraints...")

        original_count = pivot.dataset_meta.row_count
        current_count = len(df)
        loss_pct = (original_count - current_count) / original_count

        if loss_pct > 0.05:
            print(f"⚠️ [Auditor] ALERT! Data loss {loss_pct*100}% exceeds 5% constraint.")
            pivot.status = "failed"
        else:
            print("✅ [Auditor] Constraints passed.")
            if pivot.known_issues:
                pivot.status = "planning"
            else:
                pivot.status = "scanning"

        return pivot


# ==========================================
# 4. ORCHESTRATION LOOP
# ==========================================


def run_stella_loop(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataCleansingPivot]:
    """Run the full scan-plan-act-audit loop on the provided dataframe."""

    initial_meta = DatasetEntity(name="raw_data", row_count=len(df), columns={})
    pivot = DataCleansingPivot(pivot_id="job_001", dataset_meta=initial_meta)

    scanner = ScannerNet()
    strategist = StrategistNet()
    surgeon = SurgeonNet()
    auditor = AuditorNet()

    steps = 0
    max_steps = 10

    print(f"🚀 STELLA Initialized for {len(df)} rows.\n")

    while pivot.status not in {"completed", "failed"} and steps < max_steps:
        steps += 1
        print(f"\n--- Cycle {steps} [State: {pivot.status}] ---")

        if pivot.status == "scanning":
            pivot = scanner.run(df, pivot)

        elif pivot.status == "planning":
            pivot = strategist.run(pivot)

        elif pivot.status == "executing":
            df, pivot = surgeon.run(df, pivot)

        elif pivot.status == "auditing":
            pivot = auditor.run(df, pivot)

    if pivot.status == "completed":
        print("\n✨ Data Cleansing Completed Successfully!")
        print("Dataset Info:")
        print(df.info())
    else:
        print("\n💀 Process Failed or Timed Out.")

    return df, pivot


# ==========================================
# 5. DEMO EXECUTION
# ==========================================


def main() -> None:
    data = {
        "customer_id": [1, 2, 3, 4, 5],
        "tenure": [12, 24, np.nan, 8, 15],
        "total_charges": ["100.50", "200.00", " ", "450.20", "120.00"],
    }
    df_dirty = pd.DataFrame(data)

    df_clean, final_pivot = run_stella_loop(df_dirty)

    print("\n--- Final Pivot Action Log ---")
    for log in final_pivot.action_log:
        print(f"Step {log.step_id}: {log.code_executed} -> Success: {log.success}")


if __name__ == "__main__":
    main()
