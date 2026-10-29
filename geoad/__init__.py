import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMODULES = ROOT / "sumodules"

sys.path.insert(0, str(SUBMODULES / "mmpretrain"))
sys.path.insert(0, str(SUBMODULES / "mmsegmentation"))
sys.path.insert(0, str(SUBMODULES / "open-cd"))