import os
import shutil
from typing import List, Dict, Any

class WindowsActions:
    """Windows-specific file operations"""
    def list_directory(self, path: str = '.') -> List[str]:
        try:
            return os.listdir(path)
        except Exception as e:
            raise OSError(f"Error listing directory {path}: {str(e)}")

    def create_directory(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            raise OSError(f"Error creating directory {path}: {str(e)}")

    def delete_directory(self, path: str, recursive: bool = False) -> None:
        try:
            if recursive:
                shutil.rmtree(path)
            else:
                os.rmdir(path)
        except Exception as e:
            raise OSError(f"Error deleting directory {path}: {str(e)}")

    def move_file(self, src: str, dest: str) -> None:
        try:
            shutil.move(src, dest)
        except Exception as e:
            raise OSError(f"Error moving file from {src} to {dest}: {str(e)}")

    def copy_file(self, src: str, dest: str) -> None:
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            raise OSError(f"Error copying file from {src} to {dest}: {str(e)}")

    def delete_file(self, path: str) -> None:
        try:
            os.remove(path)
        except Exception as e:
            raise OSError(f"Error deleting file {path}: {str(e)}")

    def get_file_size(self, path: str) -> int:
        try:
            return os.path.getsize(path)
        except Exception as e:
            raise OSError(f"Error getting size of file {path}: {str(e)}")

    def get_directory_size(self, path: str) -> int:
        total_size = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            return total_size
        except Exception as e:
            raise OSError(f"Error getting size of directory {path}: {str(e)}")
