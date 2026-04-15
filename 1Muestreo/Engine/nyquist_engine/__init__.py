import sys
from pathlib import Path

# Add the compiled build output directory to sys.path
_engine_dir = Path(__file__).parent.parent  # Engine/
_project_root = _engine_dir.parent
_build_dir = _project_root / "build" / "lib.win-amd64-cpython-313" / "nyquist_engine"

_build_dir = _project_root / "Engine" / "nyquist_engine"

if str(_build_dir) not in sys.path:
    sys.path.insert(0, str(_build_dir))

from nyquist_core import compute_nyquist_simulation