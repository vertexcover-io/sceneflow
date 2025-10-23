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

__version__ = "0.1.0"

__all__ = [
    'CutPointRanker',
    'RankingConfig',
    'FrameFeatures',
    'FrameScore',
    'RankedFrame',
    'get_cut_frame',
    'get_ranked_cut_frames',
    '__version__'
]
