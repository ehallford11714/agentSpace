from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from ..utils.logging import get_logger

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the base agent
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.state = {}
        self.tools = {}
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize agent-specific resources"""
        pass
    
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task
        
        Args:
            task (Dict[str, Any]): Task to execute
            
        Returns:
            Any: Task execution result
        """
        pass
    
    def add_tool(self, tool_name: str, tool_instance: Any) -> None:
        """
        Add a tool to the agent's toolset
        
        Args:
            tool_name (str): Name of the tool
            tool_instance (Any): Tool instance
        """
        self.tools[tool_name] = tool_instance
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name
        
        Args:
            tool_name (str): Name of the tool
            
        Returns:
            Optional[Any]: Tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update agent's state
        
        Args:
            state (Dict[str, Any]): New state
        """
        self.state.update(state)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return self.state.copy()
    
    def log(self, message: str, level: str = 'info') -> None:
        """
        Log a message
        
        Args:
            message (str): Message to log
            level (str): Log level (info, warning, error)
        """
        log_method = getattr(self.logger, level.lower())
        log_method(message)
    
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate a task
        
        Args:
            task (Dict[str, Any]): Task to validate
            
        Returns:
            bool: True if task is valid, False otherwise
        """
        required_fields = ['name', 'type', 'parameters']
        return all(field in task for field in required_fields)
    
    def handle_error(self, error: Exception) -> None:
        """
        Handle errors during task execution
        
        Args:
            error (Exception): The error that occurred
        """
        self.logger.error(f"Error during task execution: {str(error)}")
        self.state['last_error'] = {
            'timestamp': datetime.now().isoformat(),
            'message': str(error)
        }
