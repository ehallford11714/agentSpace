from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
from utils.logging import get_logger
from toolLib.tool_registry import ToolRegistry

class Task(ABC):
    """Base class for workflow tasks"""
    
    def __init__(self, name: str, tool: str, parameters: Dict[str, Any], dependencies: Optional[List[str]] = None):
        """
        Initialize a task
        
        Args:
            name (str): Task name
            tool (str): Tool to use for task execution
            parameters (Dict[str, Any]): Task parameters
            dependencies (Optional[List[str]]): List of dependent task names
        """
        self.name = name
        self.tool = tool
        self.parameters = parameters
        self.dependencies = dependencies or []
        self.logger = get_logger(f'Task.{name}')
        self.state = {
            'status': 'pending',
            'start_time': None,
            'end_time': None,
            'result': None,
            'error': None
        }
    
    @abstractmethod
    def execute(self, tool_registry: 'ToolRegistry', workflow_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the task

        Args:
            tool_registry (ToolRegistry): Registry of available tools
            workflow_context (Optional[Dict[str, Any]]): Context of workflow execution for dependent data

        Returns:
            Dict[str, Any]: Task execution result
            
        Raises:
            Exception: If task execution fails
        """
        pass
    
    def validate_dependencies(self, workflow: 'Workflow') -> bool:
        """
        Validate task dependencies
        
        Args:
            workflow (Workflow): Parent workflow
            
        Returns:
            bool: True if all dependencies are satisfied, False otherwise
        """
        for dep_name in self.dependencies:
            dep_task = next((t for t in workflow.tasks if t.name == dep_name), None)
            if not dep_task or dep_task.state['status'] != 'completed':
                return False
        return True
    
    def update_state(self, status: str, result: Optional[Any] = None, error: Optional[Exception] = None) -> None:
        """
        Update task state
        
        Args:
            status (str): New status
            result (Optional[Any]): Task result
            error (Optional[Exception]): Error if any
        """
        now = datetime.now().isoformat()
        self.state['status'] = status

        if status == 'running':
            self.state['start_time'] = now

        if status in {'completed', 'failed', 'blocked', 'skipped'}:
            self.state['end_time'] = now

        if result is not None:
            self.state['result'] = result
        if error is not None:
            self.state['error'] = str(error)
    
    def __str__(self) -> str:
        """String representation of the task"""
        return f"Task(name={self.name}, tool={self.tool}, status={self.state['status']})"


class RegisteredToolTask(Task):
    """Concrete task that executes a registered tool from the ToolRegistry."""

    def execute(self, tool_registry: 'ToolRegistry', workflow_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.update_state('running')
        try:
            tool_class = tool_registry.get_tool_class(self.tool)
            if not tool_class:
                raise ValueError(f"Tool '{self.tool}' is not registered")

            tool_instance = tool_registry.get_tool_instance(self.tool) or tool_registry.create_tool_instance(
                self.tool, self.parameters.get('config', {})
            )

            execution_payload = {**self.parameters, 'context': workflow_context or {}, 'task_name': self.name}

            if hasattr(tool_instance, 'validate') and not tool_instance.validate(execution_payload):
                raise ValueError(f"Validation failed for tool '{self.tool}' with parameters {execution_payload}")

            result = tool_instance.execute(execution_payload)
            self.update_state('completed', result=result)
        except Exception as exc:  # pragma: no cover - defensive logging wrapper
            self.update_state('failed', error=exc)
            self.logger.error("Task execution failed", error=str(exc), task=self.name)
            raise

        self.logger.info("Task completed", task=self.name, result=self.state['result'])
        return self.state
