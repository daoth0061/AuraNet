"""
Configuration utilities for AuraNet
Provides safe YAML loading with type validation
"""

import yaml
import os
from typing import Dict, Any, Union
import os
import sys

# Add the project root directory to Python path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)



def safe_load_config(config_path: str) -> Dict[str, Any]:
    """
    Safely load YAML config with type validation.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        config: Dictionary with properly typed values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate and fix common type issues
    config = validate_config_types(config)
    
    return config


def validate_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix common type issues in config.
    
    Args:
        config: Raw config dictionary
        
    Returns:
        config: Config with validated types
    """
    if not isinstance(config, dict):
        return config
    
    # Recursively process nested dictionaries
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = validate_config_types(value)
        elif isinstance(value, str):
            # Try to convert string representations of numbers
            config[key] = try_convert_numeric_string(value)
        elif isinstance(value, list):
            # Process lists
            config[key] = [validate_config_types(item) if isinstance(item, dict) 
                          else try_convert_numeric_string(item) if isinstance(item, str) 
                          else item for item in value]
    
    return config


def try_convert_numeric_string(value: str) -> Union[str, float, int]:
    """
    Try to convert string to appropriate numeric type.
    
    Args:
        value: String value to convert
        
    Returns:
        Converted value or original string if conversion fails
    """
    if not isinstance(value, str):
        return value
        
    # Remove whitespace
    value = value.strip()
    
    # Skip if it's clearly not a number
    if not value or any(char.isalpha() for char in value.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '')):
        return value
    
    try:
        # Try scientific notation first
        if 'e' in value.lower():
            return float(value)
        
        # Try integer
        if '.' not in value:
            return int(value)
        
        # Try float
        return float(value)
        
    except (ValueError, TypeError):
        return value


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Args:
        config: Config dictionary
        key_path: Dot-separated key path (e.g., 'training.pretrain.learning_rate')
        default: Default value if key not found
        
    Returns:
        Config value or default
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set nested config value using dot notation.
    
    Args:
        config: Config dictionary
        key_path: Dot-separated key path
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


# Backward compatibility functions
def load_config_safe(config_path_or_dict: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load config with backward compatibility.
    
    Args:
        config_path_or_dict: Either path to config file or config dictionary
        
    Returns:
        Validated config dictionary
    """
    if isinstance(config_path_or_dict, str):
        return safe_load_config(config_path_or_dict)
    elif isinstance(config_path_or_dict, dict):
        return validate_config_types(config_path_or_dict)
    else:
        raise ValueError(f"Invalid config type: {type(config_path_or_dict)}")
