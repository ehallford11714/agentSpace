import os
import platform
import shutil
from typing import List, Dict, Optional
import pathlib

class OSAction:
    def __init__(self):
        """
        Initialize the OSAction class with OS-specific implementations
        """
        self.os_type = platform.system()
        self.os_actions = {
            'Windows': WindowsActions(),
            'Linux': UnixActions(),
            'Darwin': UnixActions()  # macOS is Unix-based
        }
        self.current_action = self.os_actions.get(self.os_type, None)
        
        if self.current_action is None:
            raise NotImplementedError(f"OS type {self.os_type} not supported")

    def list_directory(self, path: str = '.') -> List[str]:
        """
        List contents of a directory
        
        Args:
            path (str): Directory path to list. Defaults to current directory
            
        Returns:
            List[str]: List of directory contents
        """
        return self.current_action.list_directory(path)

    def create_directory(self, path: str) -> None:
        """
        Create a new directory
        
        Args:
            path (str): Path of the new directory
        """
        self.current_action.create_directory(path)

    def delete_directory(self, path: str, recursive: bool = False) -> None:
        """
        Delete a directory
        
        Args:
            path (str): Path of the directory to delete
            recursive (bool): If True, deletes directory and all contents
        """
        self.current_action.delete_directory(path, recursive)

    def move_file(self, src: str, dest: str) -> None:
        """
        Move a file from source to destination
        
        Args:
            src (str): Source file path
            dest (str): Destination file path
        """
        self.current_action.move_file(src, dest)

    def copy_file(self, src: str, dest: str) -> None:
        """
        Copy a file from source to destination
        
        Args:
            src (str): Source file path
            dest (str): Destination file path
        """
        self.current_action.copy_file(src, dest)

    def delete_file(self, path: str) -> None:
        """
        Delete a file
        
        Args:
            path (str): Path of the file to delete
        """
        self.current_action.delete_file(path)

    def get_file_size(self, path: str) -> int:
        """
        Get size of a file in bytes
        
        Args:
            path (str): Path of the file
            
        Returns:
            int: Size of the file in bytes
        """
        return self.current_action.get_file_size(path)

    def get_directory_size(self, path: str) -> int:
        """
        Get total size of a directory in bytes
        
        Args:
            path (str): Path of the directory
            
        Returns:
            int: Total size of directory in bytes
        """
        return self.current_action.get_directory_size(path)

    def check_file_exists(self, path: str) -> bool:
        """
        Check if a file exists
        
        Args:
            path (str): Path to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        return os.path.exists(path)

    def check_is_directory(self, path: str) -> bool:
        """
        Check if a path is a directory
        
        Args:
            path (str): Path to check
            
        Returns:
            bool: True if path is directory, False otherwise
        """
        return os.path.isdir(path)

    def check_is_file(self, path: str) -> bool:
        """
        Check if a path is a file
        
        Args:
            path (str): Path to check
            
        Returns:
            bool: True if path is file, False otherwise
        """
        return os.path.isfile(path)

    def get_file_extension(self, path: str) -> str:
        """
        Get the file extension
        
        Args:
            path (str): Path of the file
            
        Returns:
            str: File extension (including leading .)
        """
        return os.path.splitext(path)[1]

    def get_file_permissions(self, path: str) -> str:
        """
        Get file permissions in octal format
        
        Args:
            path (str): Path of the file
            
        Returns:
            str: File permissions in octal format
        """
        try:
            return oct(os.stat(path).st_mode)[-3:]
        except Exception as e:
            raise OSError(f"Error getting permissions for {path}: {str(e)}")

    def set_file_permissions(self, path: str, permissions: str) -> None:
        """
        Set file permissions
        
        Args:
            path (str): Path of the file
            permissions (str): Permissions in octal format (e.g., '755')
        """
        try:
            os.chmod(path, int(permissions, 8))
        except Exception as e:
            raise OSError(f"Error setting permissions for {path}: {str(e)}")

    def get_file_creation_time(self, path: str) -> float:
        """
        Get file creation time
        
        Args:
            path (str): Path of the file
            
        Returns:
            float: Creation time in seconds since epoch
        """
        try:
            return os.path.getctime(path)
        except Exception as e:
            raise OSError(f"Error getting creation time for {path}: {str(e)}")

    def get_file_modification_time(self, path: str) -> float:
        """
        Get file modification time
        
        Args:
            path (str): Path of the file
            
        Returns:
            float: Modification time in seconds since epoch
        """
        try:
            return os.path.getmtime(path)
        except Exception as e:
            raise OSError(f"Error getting modification time for {path}: {str(e)}")

    def get_file_access_time(self, path: str) -> float:
        """
        Get file access time
        
        Args:
            path (str): Path of the file
            
        Returns:
            float: Access time in seconds since epoch
        """
        try:
            return os.path.getatime(path)
        except Exception as e:
            raise OSError(f"Error getting access time for {path}: {str(e)}")

    def create_symbolic_link(self, target: str, link_name: str) -> None:
        """
        Create a symbolic link
        
        Args:
            target (str): Target path that the link will point to
            link_name (str): Name of the symbolic link to create
        """
        try:
            if self.os_type == 'Windows':
                os.symlink(target, link_name, target_is_directory=os.path.isdir(target))
            else:
                os.symlink(target, link_name)
        except Exception as e:
            raise OSError(f"Error creating symbolic link: {str(e)}")

    def read_file(self, path: str, encoding: str = 'utf-8') -> str:
        """
        Read contents of a file
        
        Args:
            path (str): Path of the file to read
            encoding (str): File encoding. Defaults to 'utf-8'
            
        Returns:
            str: File contents
        """
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            raise OSError(f"Error reading file {path}: {str(e)}")

    def write_file(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """
        Write content to a file
        
        Args:
            path (str): Path of the file to write
            content (str): Content to write
            encoding (str): File encoding. Defaults to 'utf-8'
        """
        try:
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
        except Exception as e:
            raise OSError(f"Error writing to file {path}: {str(e)}")

    def append_to_file(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """
        Append content to a file
        
        Args:
            path (str): Path of the file to append to
            content (str): Content to append
            encoding (str): File encoding. Defaults to 'utf-8'
        """
        try:
            with open(path, 'a', encoding=encoding) as f:
                f.write(content)
        except Exception as e:
            raise OSError(f"Error appending to file {path}: {str(e)}")

    def get_disk_usage(self, path: str = '.') -> Dict[str, int]:
        """
        Get disk usage statistics
        
        Args:
            path (str): Path to get disk usage for. Defaults to current directory
            
        Returns:
            Dict[str, int]: Dictionary containing total, used, and free space in bytes
        """
        try:
            total, used, free = shutil.disk_usage(path)
            return {
                'total': total,
                'used': used,
                'free': free
            }
        except Exception as e:
            raise OSError(f"Error getting disk usage: {str(e)}")

    def get_environment_variable(self, name: str) -> Optional[str]:
        """
        Get value of an environment variable
        
        Args:
            name (str): Name of the environment variable
            
        Returns:
            Optional[str]: Value of the environment variable, or None if not found
        """
        return os.getenv(name)

    def set_environment_variable(self, name: str, value: str) -> None:
        """
        Set an environment variable
        
        Args:
            name (str): Name of the environment variable
            value (str): Value to set
        """
        os.environ[name] = value
