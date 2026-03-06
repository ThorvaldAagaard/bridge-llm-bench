"""File utility functions for Bridge LLM Bench."""

import json
from pathlib import Path
from typing import Dict


def append_to_jsonl(file_path: Path, data: Dict):
    """
    Append a dictionary as a new line in a JSONL file.
    
    Parameters
    ----------
    file_path : Path
        Path to the JSONL file.
    data : Dict
        Dictionary to append to the file.
    
    Notes
    -----
    Creates the file if it doesn't exist.
    """
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")