"""Frame scoring for podcast/talking head cut point detection."""

import logging
import math
from typing import List, Dict

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.constants import EAR, MAR
from sceneflow.shared.models import FrameFeatures, FrameScore

logger = logging.getLogger(__name__)


class FrameScorer:
    """
    Scorer for podcast/talking head videos.
    
    Scoring logic:
    - Eye score: Penalize blinks (low EAR) and wide eyes (high EAR)
    - Mouth score: Penalize open mouth (high MAR)
    - Sharpness score: Penalize blurry/motion blur frames
    - Consistency score: Penalize frames with sudden face movement
    """

    def __init__(self, config: RankingConfig):
        self.config = config

    def compute_scores(self, features: List[FrameFeatures]) -> List[FrameScore]:
        """
        Compute scores for all frames.
        
        Args:
            features: List of FrameFeatures with all extracted metrics
            
        Returns:
            List of FrameScore
        """
        if not features:
            return []

        # Pre-compute consistency scores (needs neighbor info)
        consistency_scores = self._compute_consistency_scores(features)
        
        # Normalize sharpness scores
        sharpness_scores = self._compute_sharpness_scores(features)

        scores = []
        for i, feat in enumerate(features):
            # Eye score
            eye_score = self._score_eye(feat.eye_openness)
            
            # Mouth score
            mouth_score = self._score_mouth(feat.mouth_openness)
            
            # Sharpness score
            sharpness_score = sharpness_scores.get(feat.frame_index, 0.5)
            
            # Consistency score
            consistency_score = consistency_scores.get(feat.frame_index, 0.5)
            
            # Penalize frames with no face detected
            if not feat.face_detected:
                eye_score = 0.0
                mouth_score = 0.0
                sharpness_score = 0.0
                consistency_score = 0.0
            
            # Weighted combination
            final_score = (
                self.config.eye_weight * eye_score +
                self.config.mouth_weight * mouth_score +
                self.config.sharpness_weight * sharpness_score +
                self.config.consistency_weight * consistency_score
            )
            
            scores.append(FrameScore(
                frame_index=feat.frame_index,
                timestamp=feat.timestamp,
                eye_score=eye_score,
                mouth_score=mouth_score,
                final_score=final_score,
                # Legacy fields for backward compatibility
                eye_openness_score=eye_score,
                expression_neutrality_score=mouth_score,
                visual_sharpness_score=sharpness_score,
                motion_stability_score=consistency_score,
                composite_score=final_score,
                context_score=final_score,
            ))

        logger.info(
            "Scored %d frames: best=%.3f, worst=%.3f",
            len(scores),
            max(s.final_score for s in scores),
            min(s.final_score for s in scores)
        )

        return scores

    def _compute_sharpness_scores(self, features: List[FrameFeatures]) -> Dict[int, float]:
        """
        Normalize sharpness values to 0-1 scores.
        Uses min-max normalization with a floor threshold.
        """
        sharpness_values = [f.sharpness for f in features if f.face_detected]
        
        if not sharpness_values:
            return {f.frame_index: 0.0 for f in features}
        
        min_sharp = self.config.min_sharpness
        max_sharp = max(sharpness_values)
        
        if max_sharp <= min_sharp:
            return {f.frame_index: 1.0 for f in features}
        
        scores = {}
        for f in features:
            if f.sharpness < min_sharp:
                scores[f.frame_index] = 0.0
            else:
                scores[f.frame_index] = min(1.0, (f.sharpness - min_sharp) / (max_sharp - min_sharp))
        
        return scores

    def _compute_consistency_scores(self, features: List[FrameFeatures]) -> Dict[int, float]:
        """
        Score frame consistency based on face position stability.
        Compares each frame to its neighbors - penalizes sudden jumps.
        """
        if len(features) < 3:
            return {f.frame_index: 1.0 for f in features}
        
        scores = {}
        max_delta = self.config.max_position_delta
        
        for i, feat in enumerate(features):
            if not feat.face_detected:
                scores[feat.frame_index] = 0.0
                continue
            
            # Compare to previous and next frames
            deltas = []
            
            if i > 0 and features[i-1].face_detected:
                prev_center = features[i-1].face_center
                delta = math.sqrt(
                    (feat.face_center[0] - prev_center[0])**2 +
                    (feat.face_center[1] - prev_center[1])**2
                )
                deltas.append(delta)
            
            if i < len(features) - 1 and features[i+1].face_detected:
                next_center = features[i+1].face_center
                delta = math.sqrt(
                    (feat.face_center[0] - next_center[0])**2 +
                    (feat.face_center[1] - next_center[1])**2
                )
                deltas.append(delta)
            
            if not deltas:
                scores[feat.frame_index] = 0.5
                continue
            
            avg_delta = sum(deltas) / len(deltas)
            
            # Score: 1.0 if no movement, decreases as movement increases
            if avg_delta <= 5:  # Small movement tolerance
                scores[feat.frame_index] = 1.0
            elif avg_delta >= max_delta:
                scores[feat.frame_index] = 0.0
            else:
                scores[feat.frame_index] = 1.0 - (avg_delta - 5) / (max_delta - 5)
        
        return scores

    def _score_eye(self, ear: float) -> float:
        """
        Score eye openness - ONLY accept frames with eyes fully open.

        - EAR 0.25-0.32: Eyes fully open (BEST) -> score 1.0
        - All other values: REJECTED -> score 0.0

        This ensures only frames with properly open eyes are considered.
        """
        # Reject invalid values
        if ear < EAR.MIN_VALID or ear > EAR.MAX_VALID:
            return 0.0

        # Only accept eyes fully open in normal range
        if EAR.NORMAL_MIN <= ear <= EAR.NORMAL_MAX:
            return 1.0

        # Reject everything else (squinting, blinking, too wide)
        return 0.0

    def _score_mouth(self, mar: float) -> float:
        """
        Score mouth openness - ONLY accept frames with mouth closed.

        - MAR 0.20-0.35: Mouth closed (BEST) -> score 1.0
        - All other values: REJECTED -> score 0.0

        This ensures only frames with closed mouth are considered.
        """
        # Reject invalid values
        if mar < MAR.MIN_VALID or mar > MAR.MAX_VALID:
            return 0.0

        # Only accept closed mouth range
        if MAR.CLOSED_MIN <= mar <= MAR.CLOSED_MAX:
            return 1.0

        # Reject everything else (slightly open, talking, yawning)
        return 0.0
