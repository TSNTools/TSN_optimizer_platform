from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def start_training_process(
    command: List[str],
    log_path: str | Path,
    cwd: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    log_file = open(log_path, "ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=cwd or None,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return process


def process_status(process: Optional[subprocess.Popen]) -> str:
    if process is None:
        return "idle"
    code = process.poll()
    if code is None:
        return "running"
    if code == 0:
        return "finished"
    return f"failed ({code})"


def stop_process(process: Optional[subprocess.Popen]) -> bool:
    if process is None:
        return False
    if process.poll() is not None:
        return False
    process.terminate()
    return True
