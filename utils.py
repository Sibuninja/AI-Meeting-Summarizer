# utils.py
import os
from datetime import datetime

def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def save_text(folder: str, base: str, content: str) -> str:
    ensure_dir(folder)
    path = os.path.join(folder, f"{base}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
