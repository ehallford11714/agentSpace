"""Hugging Face dataset retrieval agent using a STELLA-like loop.

This example shows how a small set of cooperating agents can discover the
metadata for a public Hugging Face dataset, plan how to retrieve the desired
split, execute the download, and audit the result. The agents share a pivot
object that acts as the single source of truth for decisions and logs.

Usage:
    python examples/hf_dataset_agent.py                     # uses Heart Disease train split
    python examples/hf_dataset_agent.py ag_news None train  # different dataset and split

Dependencies:
    pip install datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, get_dataset_config_names, get_dataset_split_names, load_dataset
from pydantic import BaseModel, ConfigDict, Field
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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


class CorrelationEntry(BaseModel):
    pair: str
    pearson: float


class EDAReport(BaseModel):
    descriptive_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    missing_counts: Dict[str, int] = Field(default_factory=dict)
    correlations: List[CorrelationEntry] = Field(default_factory=list)
    distributions: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    insights: List[str] = Field(default_factory=list)


class CleaningReport(BaseModel):
    operations: List[str] = Field(default_factory=list)
    remaining_missing: Dict[str, int] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    name: str
    metric_name: str
    metric_value: float
    notes: List[str] = Field(default_factory=list)


class ModelReport(BaseModel):
    model_type: str
    target_column: str
    feature_columns: List[str]
    metric_name: str
    metric_value: float
    leaderboard: List[CandidateScore] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class DatasetPivot(BaseModel):
    """Single source of truth for the dataset retrieval agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pivot_id: str
    request: DatasetRequest
    available_configs: List[str] = Field(default_factory=list)
    available_splits: Dict[str, List[str]] = Field(default_factory=dict)

    dataset_obj: Optional[Dataset] = Field(default=None, exclude=True)
    raw_dataframe: Optional[pd.DataFrame] = Field(default=None, exclude=True)
    cleaned_dataframe: Optional[pd.DataFrame] = Field(default=None, exclude=True)
    dataset: Optional[DatasetSnapshot] = None
    eda_report: Optional[EDAReport] = None
    cleaning_report: Optional[CleaningReport] = None
    model_report: Optional[ModelReport] = None
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
        elif issue.issue_type == "eda_pending":
            description = "Run exploratory data analysis on the retrieved split."
            action = "run_eda"
        elif issue.issue_type == "cleaning_pending":
            description = "Cleanse the dataset with type coercion and imputation."
            action = "run_cleaning"
        elif issue.issue_type == "modeling_pending":
            description = "Train a simple model with a train/test split and report accuracy."
            action = "run_modeling"
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

    def run(self, pivot: DatasetPivot, dataset: Optional[Dataset]) -> Tuple[Optional[Dataset], DatasetPivot]:
        plan = pivot.current_plan
        if plan is None:
            raise ValueError("No plan to execute")

        dataset = dataset or pivot.dataset_obj
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
                pivot.dataset_obj = dataset
                pivot.raw_dataframe = dataset.to_pandas()
                code = "dataset = load_dataset(...); optional select(sample_size)"
                pivot.status = "auditing"
            elif plan.action_type == "run_eda":
                dataset = dataset  # carry forward existing dataset reference
                code = "trigger_eda"
                pivot.status = "analyzing"
            elif plan.action_type == "run_cleaning":
                dataset = dataset  # use existing reference
                code = "trigger_cleaning"
                pivot.status = "cleansing"
            elif plan.action_type == "run_modeling":
                dataset = dataset
                code = "trigger_modeling"
                pivot.status = "modeling"
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
        if not any(issue.issue_type == "eda_pending" for issue in pivot.known_issues):
            pivot.known_issues.append(
                Issue(
                    id="issue_eda",
                    issue_type="eda_pending",
                    severity="medium",
                    description="Exploratory analysis has not been generated yet.",
                )
            )
        pivot.status = "planning"
        return pivot


class EdaNet:
    """Analyst: compute exploratory statistics and derive insights."""

    def run(self, dataset: Dataset, pivot: DatasetPivot) -> DatasetPivot:
        print("📊 [EDA] Performing exploratory data analysis...")
        df = dataset.to_pandas()

        describe_df = df.describe(include="all").transpose()
        descriptive_stats: Dict[str, Dict[str, float]] = {
            column: {
                stat: (value if pd.notna(value) else None)
                for stat, value in stats.items()
            }
            for column, stats in describe_df.to_dict(orient="index").items()
        }

        missing_counts = df.isnull().sum().to_dict()

        correlations: List[CorrelationEntry] = []
        corr_matrix = df.select_dtypes(include=["number", "bool"]).corr(numeric_only=True)
        for i, col_i in enumerate(corr_matrix.columns):
            for col_j in corr_matrix.columns[i + 1 :]:
                value = corr_matrix.loc[col_i, col_j]
                if pd.notna(value):
                    correlations.append(CorrelationEntry(pair=f"{col_i}~{col_j}", pearson=float(value)))

        distributions: Dict[str, Dict[str, float]] = {}
        for column in df.columns:
            series = df[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                dist = series.value_counts(bins=8, normalize=True, dropna=False)
            else:
                dist = series.value_counts(normalize=True, dropna=False).head(10)
            distributions[column] = {str(bin_label): float(freq) for bin_label, freq in dist.items()}

        insights: List[str] = []
        for column, count in missing_counts.items():
            if count > 0:
                insights.append(f"Column '{column}' has {count} missing values.")

        for corr in correlations:
            if abs(corr.pearson) >= 0.5:
                insights.append(
                    f"Strong correlation detected between {corr.pair} (pearson={corr.pearson:.2f})."
                )

        for column in df.select_dtypes(include=["number", "bool"]).columns:
            skew = df[column].skew()
            if pd.notna(skew) and abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                insights.append(f"Column '{column}' shows a {direction}-skewed distribution (skew={skew:.2f}).")

        pivot.eda_report = EDAReport(
            descriptive_stats=descriptive_stats,
            missing_counts=missing_counts,
            correlations=correlations,
            distributions=distributions,
            insights=insights or ["No major data quality signals detected in EDA."],
        )

        pivot.known_issues = [issue for issue in pivot.known_issues if issue.issue_type != "eda_pending"]
        if not any(issue.issue_type == "cleaning_pending" for issue in pivot.known_issues):
            pivot.known_issues.append(
                Issue(
                    id="issue_cleaning",
                    issue_type="cleaning_pending",
                    severity="medium",
                    description="Dataset has not been cleansed for modeling.",
                )
            )
        pivot.current_plan = None
        pivot.status = "planning"

        pivot.action_log.append(
            ActionLog(
                step_id=len(pivot.action_log) + 1,
                code_executed="pandas describe + correlations + distributions",
                success=True,
                rows_affected=pivot.dataset.num_rows if pivot.dataset else 0,
            )
        )

        print("✅ [EDA] Analysis complete; insights ready in pivot.\n")
        return pivot


class CleanserNet:
    """Cleaner: apply lightweight type coercion and imputation for modeling."""

    def run(self, pivot: DatasetPivot, dataset: Dataset) -> DatasetPivot:
        print("🧹 [Cleaner] Harmonizing dataframe for modeling...")

        df = (
            pivot.cleaned_dataframe
            if pivot.cleaned_dataframe is not None
            else pivot.raw_dataframe
            if pivot.raw_dataframe is not None
            else dataset.to_pandas()
        )
        working_df = df.copy()
        operations: List[str] = []

        # Attempt numeric coercion for object columns that look numeric
        for column in working_df.columns:
            series = working_df[column]
            if pd.api.types.is_object_dtype(series):
                coerced = pd.to_numeric(series, errors="coerce")
                non_null_ratio = coerced.notnull().mean()
                if non_null_ratio >= 0.7:
                    working_df[column] = coerced
                    operations.append(
                        f"Coerced column '{column}' to numeric (kept {non_null_ratio:.0%} non-null)."
                    )

        # Impute missing values
        for column in working_df.columns:
            series = working_df[column]
            if series.isnull().any():
                if pd.api.types.is_numeric_dtype(series):
                    fill_value = series.median()
                    working_df[column].fillna(fill_value, inplace=True)
                    operations.append(f"Filled numeric missing values in '{column}' with median {fill_value}.")
                else:
                    mode = series.mode(dropna=True)
                    if not mode.empty:
                        fill_value = mode.iloc[0]
                        working_df[column].fillna(fill_value, inplace=True)
                        operations.append(f"Filled categorical missing values in '{column}' with mode '{fill_value}'.")

        pivot.cleaned_dataframe = working_df
        pivot.cleaning_report = CleaningReport(
            operations=operations or ["No cleaning operations required."],
            remaining_missing=working_df.isnull().sum().to_dict(),
        )

        pivot.known_issues = [issue for issue in pivot.known_issues if issue.issue_type != "cleaning_pending"]
        if not any(issue.issue_type == "modeling_pending" for issue in pivot.known_issues):
            pivot.known_issues.append(
                Issue(
                    id="issue_modeling",
                    issue_type="modeling_pending",
                    severity="medium",
                    description="Dataset is cleansed but not yet modeled.",
                )
            )

        pivot.current_plan = None
        pivot.status = "planning"
        pivot.action_log.append(
            ActionLog(
                step_id=len(pivot.action_log) + 1,
                code_executed="type coercion + median/mode imputation",
                success=True,
                rows_affected=len(working_df),
            )
        )

        print("✅ [Cleaner] Dataset prepared; modeling can proceed.\n")
        return pivot


class ModelBuilderNet:
    """Builder: train a baseline model with a train/test split."""

    TARGET_CANDIDATES = ["label", "labels", "target", "class"]

    def run(self, pivot: DatasetPivot, dataset: Dataset) -> DatasetPivot:
        print("🤖 [ModelBuilder] Running lightweight AutoML search...")

        df = (
            pivot.cleaned_dataframe
            if pivot.cleaned_dataframe is not None
            else pivot.raw_dataframe
            if pivot.raw_dataframe is not None
            else dataset.to_pandas()
        )
        target_column = self._choose_target(df)
        if target_column is None:
            pivot.status = "failed"
            pivot.model_report = None
            print("❌ [ModelBuilder] Could not determine a target column for modeling.")
            return pivot

        feature_columns = [col for col in df.columns if col != target_column]
        if not feature_columns:
            pivot.status = "failed"
            pivot.model_report = None
            print("❌ [ModelBuilder] No feature columns available after selecting target.")
            return pivot

        X = df[feature_columns]
        y = df[target_column]
        is_classification = self._is_classification_target(y)
        numeric_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(X[col])]
        categorical_columns = [col for col in feature_columns if col not in numeric_columns]
        text_columns = [col for col in categorical_columns if pd.api.types.is_object_dtype(X[col])]

        leaderboard: List[CandidateScore] = []
        notes: List[str] = []

        if text_columns:
            text_col = text_columns[0]
            notes.append(f"Using text column '{text_col}' for TF-IDF features.")
            y_encoded, label_encoder = self._encode_labels(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X[text_col].astype(str), y_encoded, test_size=0.2, random_state=42
            )

            text_candidates = {
                "logreg_text": LogisticRegression(max_iter=400) if is_classification else None,
                "svm_text": LinearSVC() if is_classification else None,
                "linear_reg_text": LinearSVR() if not is_classification else None,
            }

            for name, model in text_candidates.items():
                if model is None:
                    continue
                pipeline = Pipeline([
                    ("vectorizer", TfidfVectorizer()),
                    ("clf", model),
                ])
                metric_name = "accuracy" if is_classification else "r2"
                metric_value, score_notes = self._fit_and_score_text_model(
                    pipeline, X_train, X_test, y_train, y_test, metric_name
                )
                leaderboard.append(
                    CandidateScore(name=name, metric_name=metric_name, metric_value=metric_value, notes=score_notes)
                )
        if numeric_columns or categorical_columns:
            y_encoded, label_encoder = self._encode_labels(y) if is_classification else (y.astype(float), None)
            metric_name = "accuracy" if is_classification else "r2"

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_columns),
                    (
                        "cat",
                        self._one_hot_encoder(),
                        categorical_columns,
                    ),
                ]
            )

            structured_candidates = {
                "logreg": LogisticRegression(max_iter=400) if is_classification else None,
                "random_forest": self._make_random_forest_classifier() if is_classification else self._make_random_forest_regressor(),
                "linear": LinearRegression() if not is_classification else None,
            }

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            for name, model in structured_candidates.items():
                if model is None:
                    continue
                pipeline = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", model),
                ])
                metric_value, score_notes = self._fit_and_score_structured_model(
                    pipeline, X_train, X_test, y_train, y_test, metric_name, label_encoder
                )
                leaderboard.append(
                    CandidateScore(name=name, metric_name=metric_name, metric_value=metric_value, notes=score_notes)
                )

        if not leaderboard:
            pivot.status = "failed"
            pivot.model_report = None
            print("❌ [ModelBuilder] AutoML search produced no candidates.")
            return pivot

        best_candidate = max(leaderboard, key=lambda c: c.metric_value)

        pivot.model_report = ModelReport(
            model_type=best_candidate.name,
            target_column=target_column,
            feature_columns=feature_columns,
            metric_name=best_candidate.metric_name,
            metric_value=best_candidate.metric_value,
            leaderboard=leaderboard,
            notes=notes,
        )

        pivot.known_issues = [issue for issue in pivot.known_issues if issue.issue_type != "modeling_pending"]
        pivot.current_plan = None
        pivot.status = "completed"
        pivot.action_log.append(
            ActionLog(
                step_id=len(pivot.action_log) + 1,
                code_executed="train_test_split + AutoML leaderboard",
                success=True,
                rows_affected=len(df),
            )
        )

        print(
            f"✅ [ModelBuilder] Training complete. {pivot.model_report.metric_name}: "
            f"{pivot.model_report.metric_value:.3f} on held-out set.\n"
        )
        return pivot

    def _choose_target(self, df: pd.DataFrame) -> Optional[str]:
        for candidate in self.TARGET_CANDIDATES:
            if candidate in df.columns:
                return candidate
        return df.columns[-1] if len(df.columns) > 1 else None

    def _encode_labels(self, series: pd.Series) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
        if pd.api.types.is_numeric_dtype(series):
            return series.to_numpy(), None
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(series.astype(str))
        return encoded, encoder

    def _is_classification_target(self, series: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(series):
            return True
        unique_values = series.nunique(dropna=True)
        return unique_values <= 20

    def _fit_and_score_text_model(
        self,
        pipeline: Pipeline,
        X_train: pd.Series,
        X_test: pd.Series,
        y_train: np.ndarray,
        y_test: np.ndarray,
        metric_name: str,
    ) -> Tuple[float, List[str]]:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        score = (
            float(metrics.accuracy_score(y_test, preds))
            if metric_name == "accuracy"
            else float(metrics.r2_score(y_test, preds))
        )
        return score, [f"Scored {score:.3f} using {metric_name}."]

    def _fit_and_score_structured_model(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        metric_name: str,
        label_encoder: Optional[LabelEncoder],
    ) -> Tuple[float, List[str]]:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        if label_encoder:
            preds = np.round(preds).astype(int)
            preds = np.clip(preds, 0, len(label_encoder.classes_) - 1)
        score = (
            float(metrics.accuracy_score(y_test, preds))
            if metric_name == "accuracy"
            else float(metrics.r2_score(y_test, preds))
        )
        return score, [f"Scored {score:.3f} using {metric_name}."]

    def _make_random_forest_classifier(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
        )

    def _one_hot_encoder(self) -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # Backwards compatibility for older sklearn
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _make_random_forest_regressor(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
        )


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
    analyst = EdaNet()
    cleaner = CleanserNet()
    model_builder = ModelBuilderNet()

    dataset: Optional[Dataset] = None
    steps = 0
    max_steps = 18

    print(f"🚀 HF Agent initialized for repo '{request.repo_id}'.\n")

    while pivot.status not in {"completed", "failed"} and steps < max_steps:
        steps += 1
        print(f"\n--- Cycle {steps} [State: {pivot.status}] ---")

        if pivot.status == "scanning":
            pivot = scout.run(pivot)
        elif pivot.status == "planning":
            pivot = strategist.run(pivot)
        elif pivot.status == "executing":
            dataset, pivot = operator.run(pivot, dataset)
        elif pivot.status == "auditing":
            pivot = auditor.run(pivot)
        elif pivot.status == "analyzing":
            active_dataset = dataset or pivot.dataset_obj
            if active_dataset is None:
                pivot.status = "failed"
                print("❌ [EDA] No dataset available for analysis.")
            else:
                pivot = analyst.run(active_dataset, pivot)
        elif pivot.status == "cleansing":
            active_dataset = dataset or pivot.dataset_obj
            if active_dataset is None:
                pivot.status = "failed"
                print("❌ [Cleaner] No dataset available for cleansing.")
            else:
                pivot = cleaner.run(pivot, active_dataset)
        elif pivot.status == "modeling":
            active_dataset = dataset or pivot.dataset_obj
            if active_dataset is None:
                pivot.status = "failed"
                print("❌ [ModelBuilder] No dataset available for modeling.")
            else:
                pivot = model_builder.run(pivot, active_dataset)

    if pivot.status == "completed":
        print("\n✨ Dataset retrieved successfully!")
    else:
        print("\n💀 Retrieval process failed or timed out.")

    return dataset, pivot


# ==========================================
# 4. DEMO ENTRYPOINT
# ==========================================


def main(
    repo_id: str = "buio/heart-disease",
    config: Optional[str] = None,
    split: str = "train",
    sample_size: int = 400,
):
    request = DatasetRequest(repo_id=repo_id, config=config, split=split, sample_size=sample_size)
    dataset, pivot = run_stella_loop(request)

    if dataset is not None:
        print("\n--- Sample rows ---")
        print(dataset.to_pandas().head())

    if pivot.eda_report is not None:
        print("\n--- EDA Insights ---")
        for insight in pivot.eda_report.insights:
            print(f"- {insight}")

    if pivot.cleaning_report is not None:
        print("\n--- Cleaning Operations ---")
        for op in pivot.cleaning_report.operations:
            print(f"- {op}")
        print("Remaining missing values:")
        for col, count in pivot.cleaning_report.remaining_missing.items():
            print(f"  {col}: {count}")

    if pivot.model_report is not None:
        print("\n--- Model Report ---")
        print(
            f"Model: {pivot.model_report.model_type} | Target: {pivot.model_report.target_column} | "
            f"Metric ({pivot.model_report.metric_name}): {pivot.model_report.metric_value:.3f}"
        )
        if pivot.model_report.leaderboard:
            print("Leaderboard:")
            for candidate in sorted(
                pivot.model_report.leaderboard, key=lambda c: c.metric_value, reverse=True
            ):
                print(f"- {candidate.name}: {candidate.metric_value:.3f} {candidate.metric_name}")
        if pivot.model_report.notes:
            print("Notes:")
            for note in pivot.model_report.notes:
                print(f"- {note}")

    print("\n--- Action Log ---")
    for log in pivot.action_log:
        print(f"Step {log.step_id}: {log.code_executed} -> Success: {log.success}")


if __name__ == "__main__":
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else "buio/heart-disease"
    cfg = sys.argv[2] if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    sample = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    main(repo, cfg, split, sample)
