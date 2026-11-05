"""ChordVision recognition package.

Production uploads are analyzed by ``engine_adapter.ChordEngine``, which
wraps the supplied high-resolution CQT/NNLS/Viterbi recognizer and preserves
ChordVision's established result schema. The earlier self-contained engine
modules remain available for isolated tests and supporting audio/theory
helpers; the production job queue no longer calls the legacy analyzer.
"""

__version__ = "2.0.0"
ENGINE_VERSION = __version__

from .types import ChordEvent, KeyEstimate, AnalysisResult, Segment
from .analyze import analyze_file, analyze_samples, ChordEngine
from .engine_adapter import ChordEngine as ProductionChordEngine, EngineConfig as ProductionEngineConfig

__all__ = [
    "ChordEvent",
    "KeyEstimate",
    "AnalysisResult",
    "Segment",
    "analyze_file",
    "analyze_samples",
    "ChordEngine",
    "__version__",
    "ENGINE_VERSION",
    "ProductionChordEngine",
    "ProductionEngineConfig",
]
