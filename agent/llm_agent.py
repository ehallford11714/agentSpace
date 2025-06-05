from agent.base_agent import BaseAgent
from openai import OpenAI
from typing import Dict, Any, Optional
from utils.logging import get_logger

class LLMTool:
    """Tool for interacting with LLMs"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM tool
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.logger = get_logger(self.__class__.__name__)
        self.client = OpenAI(api_key=config['api_key'])
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
    
    def query(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Query the LLM
        
        Args:
            prompt (str): Prompt to send to the LLM
            **kwargs: Additional parameters for the API call
            
        Returns:
            Dict[str, Any]: LLM response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                **kwargs
            )
            return response.model_dump()
        except Exception as e:
            self.logger.error(f"LLM query failed: {str(e)}")
            raise
    
    def validate(self, task: Dict[str, Any]) -> bool:
        """
        Validate a task for the LLM
        
        Args:
            task (Dict[str, Any]): Task to validate
            
        Returns:
            bool: True if task is valid, False otherwise
        """
        required_fields = ['prompt', 'type']
        return all(field in task for field in required_fields)

class LLMAgent(BaseAgent):
    """Agent specialized for LLM operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM agent
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        super().__init__(config)
        self.llm_tool = LLMTool(config)
    
    def initialize(self) -> None:
        """Initialize LLM-specific resources"""
        # Add LLM tool to agent
        self.add_tool('llm', self.llm_tool)
    
    def execute(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task using the LLM
        
        Args:
            task (Dict[str, Any]): Task to execute
            
        Returns:
            Any: Task execution result
        """
        if not self.validate_task(task):
            raise ValueError("Invalid task format")
            
        if not self.llm_tool.validate(task):
            raise ValueError("Invalid LLM task")
            
        try:
            result = self.llm_tool.query(task['prompt'])
            return result
        except Exception as e:
            self.handle_error(e)
            raise
