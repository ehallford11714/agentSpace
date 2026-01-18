from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict, deque
from utils.logging import get_logger
from toolLib.tool_registry import ToolRegistry
from workflow.tasks import Task

class Workflow:
    """Class for managing workflows"""
    
    def __init__(self, name: str, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize a workflow
        
        Args:
            name (str): Workflow name
        """
        self.name = name
        self.tasks: List[Task] = []
        self.logger = get_logger(f'Workflow.{name}')
        self.state = {'status': 'initialized', 'start_time': None, 'end_time': None, 'error': None}
        self.tool_registry = tool_registry or ToolRegistry()
    
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
            workflow_context: Dict[str, Any] = {'workflow': {'name': self.name}, 'tasks': {}}

            tasks_by_name = {task.name: task for task in self.tasks}
            if len(tasks_by_name) != len(self.tasks):
                raise ValueError("Duplicate task names detected in workflow")

            adjacency: Dict[str, List[str]] = defaultdict(list)
            indegree: Dict[str, int] = {name: 0 for name in tasks_by_name}

            for task in self.tasks:
                for dependency in task.dependencies:
                    if dependency not in tasks_by_name:
                        raise ValueError(f"Task '{task.name}' references missing dependency '{dependency}'")
                    adjacency[dependency].append(task.name)
                    indegree[task.name] += 1

            execution_order: List[str] = []
            queue: deque[str] = deque([name for name, degree in indegree.items() if degree == 0])
            while queue:
                current = queue.popleft()
                execution_order.append(current)
                for neighbor in adjacency[current]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)

            if len(execution_order) != len(self.tasks):
                raise ValueError("Workflow has cyclic dependencies; unable to determine execution order")

            results = {}
            for task_name in execution_order:
                task = tasks_by_name[task_name]
                self.logger.info("Executing task", task=task.name)

                if not task.validate_dependencies(self):
                    task.update_state('blocked', error=f"Dependencies for task '{task.name}' are not satisfied")
                    self.state['status'] = 'failed'
                    self.state['error'] = task.state['error']
                    break

                try:
                    task_result = task.execute(self.tool_registry, workflow_context)
                    results[task.name] = task_result
                    workflow_context['tasks'][task.name] = task_result
                    self.state[f'task_{task.name}'] = task_result
                except Exception as exc:  # pragma: no cover - defensive logging wrapper
                    self.state['status'] = 'failed'
                    self.state['error'] = str(exc)
                    break

            if self.state['status'] != 'failed':
                self.state['status'] = 'completed'

            # Any tasks not executed due to a failure are marked as blocked for clarity
            if self.state['status'] == 'failed':
                for remaining in self.tasks:
                    if remaining.name not in results and remaining.state['status'] == 'pending':
                        remaining.update_state('blocked', error='Workflow stopped before execution')
                        self.state[f'task_{remaining.name}'] = remaining.state

            self.state['end_time'] = datetime.now().isoformat()

            return {
                'success': self.state['status'] == 'completed',
                'workflow': self.name,
                'status': self.state,
                'tasks': results,
                'context': workflow_context
            }

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            self.state['status'] = 'failed'
            self.state['error'] = str(e)
            self.state['end_time'] = datetime.now().isoformat()
            return {
                'success': False,
                'error': str(e),
                'workflow': self.name,
                'status': self.state
            }
