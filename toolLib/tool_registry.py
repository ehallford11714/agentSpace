from typing import Dict, Any, Type, Optional
from abc import ABC, abstractmethod
import inspect

class ToolRegistry:
    """Registry for managing tools"""
    
    def __init__(self):
        """Initialize the tool registry"""
        self._tools: Dict[str, Type] = {}
        self._tool_instances: Dict[str, Any] = {}
    
    def register_tool(self, name: str, tool_class: Type) -> None:
        """
        Register a tool class
        
        Args:
            name (str): Name of the tool
            tool_class (Type): Tool class to register
            
        Raises:
            ValueError: If tool is not a subclass of ToolBase
            ValueError: If tool name is already registered
        """
        if not issubclass(tool_class, ToolBase):
            raise ValueError(f"Tool {name} must be a subclass of ToolBase")
            
        if name in self._tools:
            raise ValueError(f"Tool {name} is already registered")
            
        self._tools[name] = tool_class
    
    def get_tool_class(self, name: str) -> Optional[Type]:
        """
        Get a tool class by name
        
        Args:
            name (str): Name of the tool
            
        Returns:
            Optional[Type]: Tool class or None if not found
        """
        return self._tools.get(name)
    
    def create_tool_instance(self, name: str, *args, **kwargs) -> Any:
        """
        Create an instance of a registered tool
        
        Args:
            name (str): Name of the tool
            *args: Arguments for tool initialization
            **kwargs: Keyword arguments for tool initialization
            
        Returns:
            Any: Tool instance
            
        Raises:
            ValueError: If tool is not registered
        """
        tool_class = self.get_tool_class(name)
        if not tool_class:
            raise ValueError(f"Tool {name} is not registered")
            
        instance = tool_class(*args, **kwargs)
        self._tool_instances[name] = instance
        return instance
    
    def get_tool_instance(self, name: str) -> Optional[Any]:
        """
        Get an existing tool instance
        
        Args:
            name (str): Name of the tool
            
        Returns:
            Optional[Any]: Tool instance or None if not found
        """
        return self._tool_instances.get(name)
    
    def validate_tool(self, tool_class: Type) -> bool:
        """
        Validate that a tool class meets requirements
        
        Args:
            tool_class (Type): Tool class to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not inspect.isclass(tool_class):
            return False
            
        if not issubclass(tool_class, ToolBase):
            return False
            
        required_methods = ['execute', 'validate']
        return all(hasattr(tool_class, method) for method in required_methods)

class ToolBase(ABC):
    """Base class for all tools"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tool
        
        Args:
            config (Dict[str, Any]): Tool configuration
        """
        self.config = config
    
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
    
    @abstractmethod
    def validate(self, task: Dict[str, Any]) -> bool:
        """
        Validate a task
        
        Args:
            task (Dict[str, Any]): Task to validate
            
        Returns:
            bool: True if task is valid, False otherwise
        """
        pass
