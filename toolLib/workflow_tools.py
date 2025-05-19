import networkx as nx
from celery import Celery
from typing import Dict, List, Optional

class WorkflowManager:
    """Manager for task workflows and dependencies"""
    def __init__(self):
        """Initialize the workflow manager"""
        self.graph = nx.DiGraph()
        self.celery_app = Celery()

    def add_task(self, task_name: str, dependencies: List[str] = None) -> None:
        """
        Add a task to the workflow
        
        Args:
            task_name (str): Name of the task
            dependencies (List[str]): List of task dependencies
        """
        self.graph.add_node(task_name)
        if dependencies:
            for dep in dependencies:
                self.graph.add_edge(dep, task_name)

    def validate_workflow(self) -> bool:
        """
        Validate that the workflow is acyclic
        
        Returns:
            bool: True if workflow is valid, False otherwise
        """
        try:
            nx.find_cycle(self.graph)
            return False
        except nx.NetworkXNoCycle:
            return True

    def get_task_order(self) -> List[str]:
        """
        Get tasks in topological order
        
        Returns:
            List[str]: List of tasks in execution order
        """
        return list(nx.topological_sort(self.graph))

    def create_celery_task(self, task_name: str, func: callable) -> None:
        """
        Create a Celery task
        
        Args:
            task_name (str): Name of the task
            func (callable): Function to execute
        """
        self.celery_app.task(name=task_name)(func)
