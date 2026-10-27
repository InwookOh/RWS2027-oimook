"""Offline entry point for UPAR 2027 Track 2."""
from pathlib import Path
import sys

# Also works when ingestion imports run.py using a file spec.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from upar.inference import InferenceEngine

_ENGINE = None


def load_model() -> None:
    global _ENGINE
    if _ENGINE is None:
        # assets/ holds model.pt, or model_<i>.pt files listed in ensemble.json.
        _ENGINE = InferenceEngine(HERE / 'assets', workers=0)


def rank_gallery(sample: dict) -> dict:
    load_model()
    return _ENGINE.rank_gallery(sample)
