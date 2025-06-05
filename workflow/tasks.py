from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from utils.logging import get_logger
from toolLib.tool_registry import ToolRegistry

@dataclass
class Task(ABC):
    """Base class for workflow tasks"""

    name: str
    tool: str
    parameters: Dict[str, Any]
    dependencies: Optional[List[str]] = field(default_factory=list)
    logger: Any = field(init=False)
    state: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.logger = get_logger(f'Task.{self.name}')
        self.state = {
            'status': 'pending',
            'start_time': None,
            'end_time': None,
            'result': None,
            'error': None,
        }
    
    @abstractmethod
    def execute(self, tool_registry: 'ToolRegistry') -> Dict[str, Any]:
        """
        Execute the task
        
        Args:
            tool_registry (ToolRegistry): Registry of available tools
            
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
        self.state['status'] = status
        self.state['end_time'] = datetime.now()
        if result:
            self.state['result'] = result
        if error:
            self.state['error'] = str(error)
    
    def __str__(self) -> str:
        """String representation of the task"""
        return f"Task(name={self.name}, tool={self.tool}, status={self.state['status']})"
