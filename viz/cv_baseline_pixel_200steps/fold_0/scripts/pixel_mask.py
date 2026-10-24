"""Compatibility import for the shared training/evaluation vehicle mask."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_mask import build_mask_from_fullres, set_mask_params
