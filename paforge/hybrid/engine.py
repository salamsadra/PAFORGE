"""Optional MATLAB Engine launcher. Uses files, not large in-memory arrays."""
from pathlib import Path


def validate_with_matlab(exchange_h5, matlab_directory=None):
    """Run MATLAB-side schema validation (requires licensed MATLAB + compatible engine)."""
    try:
        import matlab.engine
    except ImportError as exc:
        raise RuntimeError("Install the matlabengine release compatible with your installed MATLAB") from exc
    # Lifecycle is explicit to prevent hidden MATLAB sessions.
    if matlab_directory is None:
        matlab_directory = Path(__file__).resolve().parent / "matlab"
    eng = matlab.engine.start_matlab()
    try:
        eng.addpath(str(Path(matlab_directory).resolve()), nargout=0)
        return bool(eng.pa_bridge_validate(str(Path(exchange_h5).resolve()), nargout=1))
    finally:
        eng.quit()
