"""Shared pytest fixtures — make v3 root importable when running pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
