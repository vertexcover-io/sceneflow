from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


@dataclass
class VideoProperties:
    """
    Video file properties and metadata.

    Contains basic information about a video file such as frame rate,
    dimensions, and duration.
    """
    fps: float                # Frames per second
    frame_count: int          # Total number of frames
    duration: float          # Duration in seconds
    width: int               # Frame width in pixels
    height: int              # Frame height in pixels


@dataclass
class AggregatedFaceMetrics:
    """
    Aggregated facial metrics from one or more faces in a frame.

    When multiple faces are detected, these metrics represent
    center-weighted averages of all detected faces.
    """
    eye_openness: float       # Aggregated Eye Aspect Ratio (EAR)
    expression_activity: float # Aggregated Mouth Aspect Ratio (MAR)
    pose_deviation: float     # Aggregated head pose deviation


@dataclass
class EnergyRefinementResult:
    """
    Result of audio energy-based speech end refinement.

    Contains the refined timestamp and detailed metadata about the
    energy analysis used to refine the VAD timestamp.
    """
    refined_timestamp: float  # Final refined timestamp in seconds
    vad_frame: int           # Original VAD frame number
    vad_timestamp: float     # Original VAD timestamp in seconds
    refined_frame: int       # Refined frame number
    energy_drop_db: float    # Energy drop detected in dB
    frames_adjusted: int     # Number of frames adjusted (positive = moved backward)
    energy_levels: Dict[int, float]  # Frame number -> dB level mapping


@dataclass
class TemporalContext:
    """
    Temporal context information for a video frame.

    Provides timing information relative to speech end and video end.
    """
    time_since_speech_end: float  # Seconds after speech ended
    time_until_video_end: float   # Seconds before video ends
    percentage_through_video: float  # Position as percentage (0-100)


@dataclass
class NormalizedScores:
    """
    Normalized component scores for a video frame.

    All scores are in [0, 1] range where higher = better.
    """
    eye_openness: float           # Eye openness score
    motion_stability: float       # Motion stability score
    expression_neutrality: float  # Expression neutrality score
    pose_stability: float         # Pose stability score
    visual_sharpness: float       # Visual sharpness score


@dataclass
class RawMeasurements:
    """
    Raw measured values extracted from a video frame.

    These are the unprocessed measurements before normalization.
    """
    eye_aspect_ratio: float      # Raw EAR value
    motion_magnitude: float      # Raw motion in pixels
    mouth_aspect_ratio: float    # Raw MAR value
    head_pose_deviation: float   # Raw pose deviation
    sharpness_variance: float    # Raw Laplacian variance


@dataclass
class FrameMetadata:
    """
    Complete metadata for a video frame used in LLM selection.

    Combines timestamp, scores, raw measurements, and temporal context
    for LLM analysis.
    """
    timestamp: float              # Frame timestamp in seconds
    overall_score: float          # Final composite score
    scores: NormalizedScores      # Normalized component scores
    raw_measurements: RawMeasurements  # Raw measured values
    temporal_context: TemporalContext  # Temporal information


@dataclass
class FaceFeatures:
    """
    Features extracted from a single face within a frame.

    Used when multiple faces are detected in a single frame to track
    per-face metrics before aggregation.
    """
    face_index: int           # Index of this face (0 = first detected)
    bbox: Tuple[float, float, float, float]  # Bounding box (x1, y1, x2, y2)
    center_distance: float    # Normalized distance from frame center (0-1)
    center_weight: float      # Weight for center-weighted averaging (0-1)

    # Per-face metrics
    eye_openness: float       # Eye Aspect Ratio (EAR)
    expression_activity: float # Facial expression activity
    pose_deviation: float     # Head pose deviation


@dataclass
class FrameFeatures:
    """
    Raw features extracted from a single video frame.

    These are the unprocessed measurements that will be normalized
    and scored by the FrameScorer.

    For multi-face frames, the top-level metrics (eye_openness, etc.) are
    center-weighted aggregates, while individual_faces contains per-face data.
    """
    frame_index: int          # Absolute frame number in video
    timestamp: float          # Timestamp in seconds

    # Aggregated visual features (center-weighted when multiple faces)
    eye_openness: float       # Eye Aspect Ratio (EAR) - normal ~0.25-0.35
    motion_magnitude: float   # Optical flow magnitude - lower is better
    expression_activity: float # Facial expression activity - lower is better
    pose_deviation: float     # Head pose deviation - lower is better
    sharpness: float         # Image sharpness (Laplacian variance) - higher is better

    # Multi-face data
    num_faces: int = 1        # Number of faces detected
    individual_faces: Optional[List[FaceFeatures]] = None  # Per-face metrics


@dataclass
class FrameScore:
    """
    Computed scores for a single frame.

    Contains both individual component scores and the final composite score
    used for ranking. All scores are in [0, 1] range where higher = better.
    """
    frame_index: int          # Absolute frame number in video
    timestamp: float          # Timestamp in seconds

    # Component scores (normalized to [0, 1], higher = better)
    eye_openness_score: float         # Score for eye openness (prefer normal)
    motion_stability_score: float     # Score for motion (prefer low motion)
    expression_neutrality_score: float # Score for expression (prefer neutral)
    pose_stability_score: float       # Score for pose (prefer stable)
    visual_sharpness_score: float     # Score for sharpness (prefer sharp)

    # Composite and modifiers
    composite_score: float    # Weighted sum of component scores
    context_score: float      # Score considering temporal context
    quality_penalty: float    # Penalty multiplier for low quality frames
    stability_boost: float    # Boost multiplier for locally stable frames

    # Final ranking score
    final_score: float        # composite * quality_penalty * stability_boost


@dataclass
class RankedFrame:
    """
    Final ranking result for a frame.

    Represents a single ranked cut point candidate with its
    position in the ranking and final score.
    """
    rank: int                 # Rank position (1 = best)
    frame_index: int          # Absolute frame number in video
    timestamp: float          # Timestamp in seconds
    score: float             # Final score (higher = better)

    def __repr__(self) -> str:
        return (f"RankedFrame(rank={self.rank}, "
                f"frame={self.frame_index}, "
                f"time={self.timestamp:.2f}s, "
                f"score={self.score:.4f})")
