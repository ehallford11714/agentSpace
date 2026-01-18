"""Demonstrates building a workflow with dependencies and tool execution."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolLib.tool_registry import ToolRegistry, ToolBase
from workflow.workflows import Workflow
from workflow.tasks import RegisteredToolTask


class LoadDocument(ToolBase):
    """Pretend to load a document and return its path and content."""

    def execute(self, task):
        doc_path = task.get("path", "./docs/report.txt")
        return {"path": doc_path, "content": f"Loaded content from {doc_path}"}

    def validate(self, task):
        return "path" in task


class SummarizeDocument(ToolBase):
    """Summarize the document loaded by a previous task."""

    def execute(self, task):
        loaded = task["context"]["tasks"]["load_document"]["result"]
        return {"summary": f"Summary for {loaded['path']}", "source_path": loaded["path"]}

    def validate(self, task):
        return "context" in task


def main():
    """
    Build a two-step workflow. The second task depends on the first and
    receives the previous result through ``workflow_context['tasks']``.
    """

    registry = ToolRegistry()
    registry.register_tool("load_document", LoadDocument)
    registry.register_tool("summarize_document", SummarizeDocument)

    workflow = Workflow("document_analysis", tool_registry=registry)

    load_task = RegisteredToolTask(
        name="load_document",
        tool="load_document",
        parameters={"config": {}, "path": "./docs/report.txt"},
    )

    summarize_task = RegisteredToolTask(
        name="summarize",
        tool="summarize_document",
        parameters={"config": {}},
        dependencies=["load_document"],
    )

    workflow.add_task(load_task)
    workflow.add_task(summarize_task)

    result = workflow.execute()
    print("Workflow complete:\n", result)


if __name__ == "__main__":
    main()
