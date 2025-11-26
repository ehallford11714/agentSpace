"""Example workflow demonstrating document ingestion and EDA profiling."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agent.eda_agent import EDAAgent


def build_sample_dataset(csv_path: str) -> None:
    data = {
        "document_id": [1, 2, 3, 4, 5],
        "page_count": [10, 8, 15, 3, 22],
        "language": ["en", "es", "en", "fr", "en"],
        "contains_tables": [True, False, True, False, True],
    }
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(csv_path, index=False)


def run_document_workflow() -> Dict[str, Any]:
    """Create a dataset, run EDA, and persist the generated report."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "documents.csv")
        build_sample_dataset(csv_path)

        agent = EDAAgent(config={"eda": {"random_state": 123}})
        agent.initialize()

        task = {
            "name": "document_eda",
            "type": "eda",
            "parameters": {
                "file_path": csv_path,
                "columns": ["document_id", "page_count", "language"],
                "sample_fraction": 0.8,
                "report_format": "json",
                "report_path": os.path.join(temp_dir, "document_eda_report.json"),
            },
        }

        result = agent.execute(task)
        print("EDA summary keys:", list(result["summary"].keys()))
        print("Report saved to:", result["report_path"])
        return result


def main() -> None:
    run_document_workflow()


if __name__ == "__main__":
    main()
