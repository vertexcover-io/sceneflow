"""
SceneFlow - Smart Video Cut Point Detection

A multi-stage ranking system for identifying optimal video cut points
in AI-generated talking head videos based on facial features, visual quality,
and temporal stability.
"""

# Main public API
from sceneflow.api import get_cut_frame, get_ranked_cut_frames, cut_video

# Shared foundational modules (re-exported for convenience)
from sceneflow.shared import (
    RankingConfig,
    FrameFeatures,
    FrameScore,
    RankedFrame
)

# Core classes
from sceneflow.core import CutPointRanker
from sceneflow.detection import SpeechDetector, EnergyRefiner

# Airtable integration (optional - only import if available)
try:
    from sceneflow.integration import AirtableUploader, upload_to_airtable
    AIRTABLE_AVAILABLE = True
except ImportError:
    AIRTABLE_AVAILABLE = False
    AirtableUploader = None
    upload_to_airtable = None

__version__ = "0.1.0"

__all__ = [
    # Main API functions
    'get_cut_frame',
    'get_ranked_cut_frames',
    'cut_video',

    # Core classes
    'CutPointRanker',
    'SpeechDetector',
    'EnergyRefiner',

    # Configuration
    'RankingConfig',

    # Data models
    'FrameFeatures',
    'FrameScore',
    'RankedFrame',

    # Airtable integration (optional)
    'AirtableUploader',
    'upload_to_airtable',
    'AIRTABLE_AVAILABLE',

    # Version
    '__version__'
]
