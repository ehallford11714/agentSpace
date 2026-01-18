"""Agent wrapper for orchestrating exploratory data analysis tasks."""
from __future__ import annotations

import os
from typing import Any, Dict

from agent.base_agent import BaseAgent
from toolLib.tool_registry import ToolRegistry
from toolLib.eda_tool import EDATool


class EDAAgent(BaseAgent):
    """Agent responsible for validating parameters and running EDA profiles."""

    def initialize(self) -> None:  # type: ignore[override]
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool("eda_tool", EDATool)
        self.add_tool("eda_tool", self.tool_registry.create_tool_instance("eda_tool", config=self.config.get("eda", {})))

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        if not self.validate_task(task):
            raise ValueError("Task does not include required keys: name, type, parameters")

        parameters = task.get("parameters", {})
        if not self._validate_eda_parameters(parameters):
            raise ValueError("Invalid EDA parameters")

        eda_tool = self.get_tool("eda_tool")
        if not eda_tool:
            raise ValueError("EDA tool is not initialized")

        self.log(f"Running EDA task '{task['name']}' on {parameters['file_path']}")
        result = eda_tool.execute(parameters)
        self.update_state({"last_result": result})
        return result

    def _validate_eda_parameters(self, parameters: Dict[str, Any]) -> bool:
        file_path = parameters.get("file_path")
        if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
            return False

        columns = parameters.get("columns")
        if columns is not None and not isinstance(columns, list):
            return False

        sample_fraction = parameters.get("sample_fraction")
        if sample_fraction is not None:
            try:
                sample_fraction = float(sample_fraction)
            except (TypeError, ValueError):
                return False
            if sample_fraction <= 0 or sample_fraction > 1:
                return False

        return True
