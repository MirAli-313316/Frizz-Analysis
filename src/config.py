"""
Configuration management for Hair Frizz Analysis Tool.

Handles loading and saving user preferences to .app_config.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AppConfig:
    """
    Manages application configuration and user preferences.
    
    Stores settings in .app_config.json in the project root.
    """
    
    DEFAULT_CONFIG = {
        'last_output_folder': './outputs',
        'last_input_folder': None,
    }
    
    def __init__(self, config_file: str = '.app_config.json'):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to config file (relative to project root)
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary of configuration values
        """
        if not self.config_file.exists():
            logger.info(f"Config file not found, using defaults: {self.config_file}")
            return self.DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded config from {self.config_file}")
                
                # Merge with defaults to ensure all keys exist
                merged = self.DEFAULT_CONFIG.copy()
                merged.update(config)
                return merged
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            logger.info("Using default configuration")
            return self.DEFAULT_CONFIG.copy()
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            logger.info("Using default configuration")
            return self.DEFAULT_CONFIG.copy()
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved config to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
            save: Whether to immediately save to file
        
        Returns:
            True if successful (and saved if requested), False otherwise
        """
        self.config[key] = value
        
        if save:
            return self.save_config()
        
        return True
    
    def get_last_output_folder(self) -> str:
        """
        Get the last used output folder.
        
        Returns:
            Path to last output folder
        """
        return self.get('last_output_folder', './outputs')
    
    def set_last_output_folder(self, folder: str) -> bool:
        """
        Save the last used output folder.
        
        Args:
            folder: Path to output folder
        
        Returns:
            True if successful, False otherwise
        """
        return self.set('last_output_folder', folder)
    
    def get_last_input_folder(self) -> Optional[str]:
        """
        Get the last used input folder.
        
        Returns:
            Path to last input folder or None
        """
        return self.get('last_input_folder')
    
    def set_last_input_folder(self, folder: str) -> bool:
        """
        Save the last used input folder.
        
        Args:
            folder: Path to input folder
        
        Returns:
            True if successful, False otherwise
        """
        return self.set('last_input_folder', folder)

