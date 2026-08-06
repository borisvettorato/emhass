"""Compatibility shim.

Training/evaluation script moved to scripts/train_pinn_thermal.py
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    target = root / "scripts" / "train_pinn_thermal.py"
    runpy.run_path(str(target), run_name="__main__")