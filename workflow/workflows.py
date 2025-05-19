from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from ..utils.logging import get_logger

class Workflow:
    """Class for managing workflows"""
    
    def __init__(self, name: str):
        """
        Initialize a workflow
        
        Args:
            name (str): Workflow name
        """
        self.name = name
        self.tasks: List[Task] = []
        self.logger = get_logger(f'Workflow.{name}')
        self.state = {'status': 'initialized', 'start_time': None, 'end_time': None}
    
    def add_task(self, task: 'Task') -> None:
        """
        Add a task to the workflow
        
        Args:
            task (Task): Task to add
        """
        self.tasks.append(task)
        self.logger.info(f"Added task {task.name} to workflow")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow
        
        Returns:
            Dict[str, Any]: Workflow execution result
        """
        try:
            self.state['status'] = 'running'
            self.state['start_time'] = datetime.now().isoformat()
            
            results = {}
            for task in self.tasks:
                self.logger.info(f"Executing task {task.name}")
                task_result = task.execute()
                results[task.name] = task_result
                
                # Update state with task results
                self.state[f'task_{task.name}'] = task_result
            
            self.state['status'] = 'completed'
            self.state['end_time'] = datetime.now().isoformat()
            
            return {
                'success': True,
                'workflow': self.name,
                'status': self.state,
                'tasks': results
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            self.state['status'] = 'failed'
            self.state['end_time'] = datetime.now().isoformat()
            return {
                'success': False,
                'error': str(e),
                'workflow': self.name,
                'status': self.state
            }
