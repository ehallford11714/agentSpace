from typing import Dict, Any, List, Optional
import logging
from ..utils.logging import get_logger
from ..toolLib.tool_registry import ToolRegistry

class Task:
    """Class for workflow tasks"""
    
    def __init__(self, name: str, tool: str, parameters: Dict[str, Any], dependencies: List[str] = None):
        """
        Initialize a task
        
        Args:
            name (str): Task name
            tool (str): Tool name
            parameters (Dict[str, Any]): Task parameters
            dependencies (List[str]): List of dependent task names
        """
        self.name = name
        self.tool = tool
        self.parameters = parameters
        self.dependencies = dependencies or []
        self.logger = get_logger(f'Task.{name}')
        self.state = {'status': 'initialized', 'start_time': None, 'end_time': None}
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the task
        
        Returns:
            Dict[str, Any]: Task execution result
        """
        try:
            self.state['status'] = 'running'
            self.state['start_time'] = datetime.now().isoformat()
            
            # Get the tool instance from the registry
            tool_instance = ToolRegistry().get_tool_instance(self.tool)
            if not tool_instance:
                raise ValueError(f"Tool {self.tool} not found")
            
            # Prepare parameters with any required substitutions
            params = self._prepare_parameters()
            
            # Execute the tool
            result = tool_instance.execute(params)
            
            self.state['status'] = 'completed'
            self.state['end_time'] = datetime.now().isoformat()
            
            return {
                'success': True,
                'task': self.name,
                'result': result,
                'status': self.state
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            self.state['status'] = 'failed'
            self.state['end_time'] = datetime.now().isoformat()
            return {
                'success': False,
                'error': str(e),
                'task': self.name,
                'status': self.state
            }
    
    def _prepare_parameters(self) -> Dict[str, Any]:
        """
        Prepare parameters with any required substitutions
        
        Returns:
            Dict[str, Any]: Prepared parameters
        """
        params = self.parameters.copy()
        
        # Handle parameter substitutions
        for key, value in params.items():
            if isinstance(value, str) and '{' in value:
                # This is a placeholder for substitution
                # In a full implementation, we would replace these with actual values
                # from previous tasks or context
                params[key] = value.format(**self._get_context())
        
        return params
    
    def _get_context(self) -> Dict[str, Any]:
        """
        Get context for parameter substitution
        
        Returns:
            Dict[str, Any]: Context dictionary
        """
        # In a full implementation, this would get context from previous tasks
        return {}
    
    def validate(self) -> bool:
        """
        Validate the task
        
        Returns:
            bool: True if task is valid, False otherwise
        """
        return all([
            self.name,
            self.tool,
            isinstance(self.parameters, dict)
        ])
