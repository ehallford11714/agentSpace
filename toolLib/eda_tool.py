"""Exploratory data analysis tool built on top of ToolBase."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .tool_registry import ToolBase


class EDATool(ToolBase):
    """Tool for performing simple exploratory data analysis tasks."""

    SUPPORTED_FORMATS = {"json", "html"}

    def validate(self, task: Dict[str, Any]) -> bool:  # type: ignore[override]
        file_path = task.get("file_path")
        if not file_path or not isinstance(file_path, str):
            return False
        if not os.path.exists(file_path):
            return False

        sample_fraction = task.get("sample_fraction")
        if sample_fraction is not None:
            try:
                sample_fraction = float(sample_fraction)
            except (TypeError, ValueError):
                return False
            if sample_fraction <= 0 or sample_fraction > 1:
                return False

        columns = task.get("columns")
        if columns is not None and not isinstance(columns, list):
            return False

        report_format = task.get("report_format", "json")
        if report_format not in self.SUPPORTED_FORMATS:
            return False
        return True

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        if not self.validate(task):
            raise ValueError("Invalid task configuration for EDA tool")

        file_path = task["file_path"]
        columns: Optional[List[str]] = task.get("columns")
        sample_fraction: Optional[float] = task.get("sample_fraction")
        report_format: str = task.get("report_format", "json")
        report_path: Optional[str] = task.get("report_path")

        dataframe = self.load_dataset(file_path, columns=columns, sample_fraction=sample_fraction)
        summary = self.compute_summary_statistics(dataframe)
        report_path = self.generate_report(summary, report_format=report_format, report_path=report_path)

        return {
            "file_path": file_path,
            "row_count": len(dataframe),
            "columns": list(dataframe.columns),
            "summary": summary,
            "report_path": report_path,
        }

    def load_dataset(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        sample_fraction: Optional[float] = None,
    ) -> pd.DataFrame:
        dataframe = pd.read_csv(file_path)
        if columns:
            missing = [col for col in columns if col not in dataframe.columns]
            if missing:
                raise ValueError(f"Columns not found in dataset: {', '.join(missing)}")
            dataframe = dataframe[columns]
        if sample_fraction:
            dataframe = dataframe.sample(frac=sample_fraction, random_state=self.config.get("random_state", 42))
        dataframe = dataframe.reset_index(drop=True)
        return dataframe

    def compute_summary_statistics(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        describe_df = dataframe.describe(include="all").fillna("null")
        return json.loads(describe_df.to_json())

    def generate_report(
        self,
        summary: Dict[str, Any],
        report_format: str = "json",
        report_path: Optional[str] = None,
    ) -> str:
        if report_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported report format: {report_format}")

        if not report_path:
            report_path = f"eda_report.{report_format}"

        if report_format == "json":
            with open(report_path, "w", encoding="utf-8") as report_file:
                json.dump(summary, report_file, indent=2)
        else:
            html_content = self._summary_to_html(summary)
            with open(report_path, "w", encoding="utf-8") as report_file:
                report_file.write(html_content)

        return report_path

    def _summary_to_html(self, summary: Dict[str, Any]) -> str:
        header = "<html><head><title>EDA Report</title></head><body><h1>Summary Statistics</h1>"
        rows = []
        for metric, values in summary.items():
            rows.append(f"<tr><th>{metric}</th><td>{values}</td></tr>")
        footer = "</body></html>"
        return f"{header}<table>{''.join(rows)}</table>{footer}"
