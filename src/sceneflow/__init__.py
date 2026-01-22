"""SceneFlow - Smart Video Cut Point Detection."""

from sceneflow.api import (
    AnalysisResult,
    run_analysis_async,
    get_cut_frames,
    get_cut_frames_async,
    cut_video,
    cut_video_async,
)

from sceneflow.shared import RankingConfig, FrameFeatures, FrameScore, RankedFrame
from sceneflow.core import CutPointRanker
from sceneflow.detection import SpeechDetector
from sceneflow.utils.video import VideoSession, cleanup_downloaded_video_async

try:
    from sceneflow.integration import (
        AirtableUploader,
        upload_to_airtable,
        analyze_and_upload_to_airtable,
        analyze_ranked_and_upload_to_airtable,
        cut_and_upload_to_airtable,
    )

    AIRTABLE_AVAILABLE = True
except ImportError:
    AIRTABLE_AVAILABLE = False
    AirtableUploader = None
    upload_to_airtable = None
    analyze_and_upload_to_airtable = None
    analyze_ranked_and_upload_to_airtable = None
    cut_and_upload_to_airtable = None

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "run_analysis_async",
    "cleanup_downloaded_video_async",
    "get_cut_frames",
    "get_cut_frames_async",
    "cut_video",
    "cut_video_async",
    "CutPointRanker",
    "SpeechDetector",
    "VideoSession",
    "RankingConfig",
    "FrameFeatures",
    "FrameScore",
    "RankedFrame",
    "AirtableUploader",
    "upload_to_airtable",
    "analyze_and_upload_to_airtable",
    "analyze_ranked_and_upload_to_airtable",
    "cut_and_upload_to_airtable",
    "AIRTABLE_AVAILABLE",
    "__version__",
]
