import json 
import os
import sys
from pathlib import Path
from typing import Any, List

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError 
from ensure import ensure_annotations

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mlops.utils import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return its contents as a ConfigBox.
    
    Args:
        path_to_yaml (Path): Path to YAML file
        
    Returns:
        ConfigBox: Contents of YAML file
        
    Raises:
        ValueError: If YAML file is empty
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
    """Create directories if they don't exist.
    
    Args:
        path_to_directories (List[Path]): List of directory paths
        verbose (bool, optional): Whether to log creation. Defaults to True
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """Save data as JSON file.
    
    Args:
        path (Path): Path to save JSON file
        data (dict): Data to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load JSON file and return contents.
    
    Args:
        path (Path): Path to JSON file
        
    Returns:
        ConfigBox: Contents of JSON file
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded from {path}")
    return ConfigBox(content)


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary file using joblib.
    
    Args:
        path (Path): Path to binary file
        
    Returns:
        Any: Contents of binary file
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Get file size in KB.
    
    Args:
        path (Path): Path to file
        
    Returns:
        str: File size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"