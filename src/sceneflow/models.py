from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameFeatures:
    """
    Raw features extracted from a single video frame.

    These are the unprocessed measurements that will be normalized
    and scored by the FrameScorer.
    """
    frame_index: int          # Absolute frame number in video
    timestamp: float          # Timestamp in seconds

    # Visual features
    eye_openness: float       # Eye Aspect Ratio (EAR) - normal ~0.25-0.35
    motion_magnitude: float   # Optical flow magnitude - lower is better
    expression_activity: float # Facial expression activity - lower is better
    pose_deviation: float     # Head pose deviation - lower is better
    sharpness: float         # Image sharpness (Laplacian variance) - higher is better


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
