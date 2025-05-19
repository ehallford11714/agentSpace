import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from ..utils.logging import get_logger

class FileTool(ToolBase):
    """Tool for file operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file tool
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        super().__init__(config)
        self.logger = get_logger(self.__class__.__name__)
        self.base_path = Path(config.get('base_path', '.'))
    
    def create_file(self, path: str, content: str = '') -> Dict[str, Any]:
        """
        Create a new file
        
        Args:
            path (str): Path to create
            content (str): Initial content
            
        Returns:
            Dict[str, Any]: Operation result
        """
        full_path = self.base_path / path
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {'success': True, 'path': str(full_path)}
        except Exception as e:
            self.logger.error(f"Failed to create file {path}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read file contents
        
        Args:
            path (str): Path to read
            
        Returns:
            Dict[str, Any]: File contents and metadata
        """
        full_path = self.base_path / path
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                'success': True,
                'content': content,
                'path': str(full_path),
                'size': full_path.stat().st_size
            }
        except Exception as e:
            self.logger.error(f"Failed to read file {path}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def delete_file(self, path: str) -> Dict[str, Any]:
        """
        Delete a file
        
        Args:
            path (str): Path to delete
            
        Returns:
            Dict[str, Any]: Operation result
        """
        full_path = self.base_path / path
        try:
            if full_path.exists():
                full_path.unlink()
            return {'success': True, 'path': str(full_path)}
        except Exception as e:
            self.logger.error(f"Failed to delete file {path}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def move_file(self, src: str, dest: str) -> Dict[str, Any]:
        """
        Move a file
        
        Args:
            src (str): Source path
            dest (str): Destination path
            
        Returns:
            Dict[str, Any]: Operation result
        """
        src_path = self.base_path / src
        dest_path = self.base_path / dest
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))
            return {
                'success': True,
                'source': str(src_path),
                'destination': str(dest_path)
            }
        except Exception as e:
            self.logger.error(f"Failed to move file {src} to {dest}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def validate(self, task: Dict[str, Any]) -> bool:
        """
        Validate a file operation task
        
        Args:
            task (Dict[str, Any]): Task to validate
            
        Returns:
            bool: True if task is valid, False otherwise
        """
        operation = task.get('operation')
        if not operation:
            return False
            
        required_fields = {
            'create': ['path'],
            'read': ['path'],
            'delete': ['path'],
            'move': ['src', 'dest']
        }
        
        return all(field in task for field in required_fields.get(operation, []))
