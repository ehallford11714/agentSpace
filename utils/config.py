import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from .logging import get_logger

class ConfigManager:
    """Manager for configuration settings"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the config manager
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                self.logger.info(f"Loaded configuration from {self.config_path}")
            else:
                self.logger.warning(f"Configuration file {self.config_path} not found")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f)
            self.logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key (str): Configuration key
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value
        
        Args:
            key (str): Configuration key
            value (Any): Value to set
        """
        self.config[key] = value
        self.save_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a configuration section
        
        Args:
            section (str): Section name
            
        Returns:
            Dict[str, Any]: Section configuration
        """
        return self.config.get(section, {})
    
    def validate(self) -> bool:
        """
        Validate the configuration
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        required_sections = ['llm', 'file_system', 'email']
        return all(section in self.config for section in required_sections)
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.get_section('llm')
    
    def get_file_system_config(self) -> Dict[str, Any]:
        """Get file system configuration"""
        return self.get_section('file_system')
    
    def get_email_config(self) -> Dict[str, Any]:
        """Get email configuration"""
        return self.get_section('email')
