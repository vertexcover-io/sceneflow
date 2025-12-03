from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple


@dataclass
class VideoProperties:
    """Video file properties."""
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int


@dataclass
class EnergyRefinementResult:
    """Result of audio energy-based speech end refinement."""
    refined_timestamp: float
    vad_frame: int
    vad_timestamp: float
    refined_frame: int
    energy_drop_db: float
    frames_adjusted: int
    energy_levels: Dict[int, float]


@dataclass
class FaceFeatures:
    """Features extracted from a single face within a frame."""
    face_index: int
    bbox: Tuple[float, float, float, float]
    center_distance: float
    center_weight: float
    eye_openness: float
    expression_activity: float
    pose_deviation: float


@dataclass
class FrameFeatures:
    """Features extracted from a video frame for cut point detection."""
    frame_index: int
    timestamp: float
    eye_openness: float      # EAR value (0.25-0.35 normal, <0.2 = blink)
    mouth_openness: float    # MAR value (<0.15 closed, >0.3 = talking)
    face_detected: bool      # Whether a face was found
    sharpness: float = 100.0  # Laplacian variance (higher = sharper)
    face_center: Tuple[float, float] = (0.0, 0.0)  # Face center for consistency check
    
    # Legacy fields for backward compatibility
    motion_magnitude: float = 0.0
    expression_activity: float = 0.0
    pose_deviation: float = 0.0
    num_faces: int = 1
    individual_faces: Optional[List[FaceFeatures]] = None


@dataclass
class FrameScore:
    """Score for a single frame."""
    frame_index: int
    timestamp: float
    eye_score: float         # 0-1, higher = eyes more open
    mouth_score: float       # 0-1, higher = mouth more closed
    final_score: float       # Weighted combination
    
    # Legacy fields for backward compatibility
    eye_openness_score: float = 0.0
    motion_stability_score: float = 1.0
    expression_neutrality_score: float = 0.0
    pose_stability_score: float = 1.0
    visual_sharpness_score: float = 1.0
    composite_score: float = 0.0
    context_score: float = 0.0
    quality_penalty: float = 1.0
    stability_boost: float = 1.0


@dataclass
class RankedFrame:
    """Ranked cut point candidate."""
    rank: int
    frame_index: int
    timestamp: float
    score: float

    def __repr__(self) -> str:
        return f"RankedFrame(rank={self.rank}, time={self.timestamp:.4f}s, score={self.score:.4f})"


# Legacy models for backward compatibility
@dataclass
class TemporalContext:
    time_since_speech_end: float
    time_until_video_end: float
    percentage_through_video: float


@dataclass
class NormalizedScores:
    eye_openness: float
    motion_stability: float
    expression_neutrality: float
    pose_stability: float
    visual_sharpness: float


@dataclass
class RawMeasurements:
    eye_aspect_ratio: float
    motion_magnitude: float
    mouth_aspect_ratio: float
    head_pose_deviation: float
    sharpness_variance: float


@dataclass
class FrameMetadata:
    timestamp: float
    overall_score: float
    scores: NormalizedScores
    raw_measurements: RawMeasurements
    temporal_context: TemporalContext


@dataclass 
class AggregatedFaceMetrics:
    eye_openness: float
    expression_activity: float
    pose_deviation: float
