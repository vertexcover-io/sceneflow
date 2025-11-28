"""
SceneFlow - Smart Video Cut Point Detection

A multi-stage ranking system for identifying optimal video cut points
in AI-generated talking head videos based on facial features, visual quality,
and temporal stability.
"""

from .ranker import CutPointRanker
from .config import RankingConfig
from .models import FrameFeatures, FrameScore, RankedFrame
from .api import get_cut_frame, get_ranked_cut_frames
from .energy_refiner import EnergyRefiner

# Airtable integration (optional - only import if available)
try:
    from .airtable_uploader import AirtableUploader, upload_to_airtable
    AIRTABLE_AVAILABLE = True
except ImportError:
    AIRTABLE_AVAILABLE = False
    AirtableUploader = None
    upload_to_airtable = None

__version__ = "0.1.0"

__all__ = [
    'CutPointRanker',
    'RankingConfig',
    'FrameFeatures',
    'FrameScore',
    'RankedFrame',
    'get_cut_frame',
    'get_ranked_cut_frames',
    'EnergyRefiner',
    'AirtableUploader',
    'upload_to_airtable',
    'AIRTABLE_AVAILABLE',
    '__version__'
]
