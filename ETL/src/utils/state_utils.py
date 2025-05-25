import os
import json
from datetime import datetime
from typing import Any

def load_state(obj: Any) -> None:
    """
    Load processing state from previous run into obj.processed_files and obj.failed_files.
    """
    try:
        if os.path.exists(obj.state_file):
            with open(obj.state_file, 'r') as f:
                state = json.load(f)
                obj.processed_files = set(state.get('processed', []))
                obj.failed_files = set(state.get('failed', []))
    except Exception as e:
        pass

def save_state(obj: Any) -> None:
    """
    Save current processing state from obj.processed_files and obj.failed_files.
    """
    try:
        with open(obj.state_file, 'w') as f:
            json.dump({
                'processed': list(obj.processed_files),
                'failed': list(obj.failed_files),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        pass

def update_state(obj: Any, file_path: str, success: bool) -> None:
    """
    Update state for a file and save.
    """
    if success:
        obj.processed_files.add(file_path)
        if file_path in obj.failed_files:
            obj.failed_files.remove(file_path)
    else:
        obj.failed_files.add(file_path)
    save_state(obj) 