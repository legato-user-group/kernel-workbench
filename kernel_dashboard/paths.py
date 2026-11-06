"""Shared storage locations; importing these never starts a worker."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('KERNEL_OPT_DATA', ROOT / '.runs'))
