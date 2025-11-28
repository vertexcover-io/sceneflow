"""Frame scoring module for cut point ranking.

This module provides the FrameScorer class which implements the multi-stage
scoring pipeline for ranking video frames based on multiple quality factors.
"""

import logging
from typing import List

import numpy as np

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.models import FrameFeatures, FrameScore
from sceneflow.core.normalizer import MetricNormalizer
from sceneflow.core.quality_gating import QualityGate
from sceneflow.core.stability_analyzer import StabilityAnalyzer

logger = logging.getLogger(__name__)


class FrameScorer:
    """
    Computes multi-factor scores for video frames.

    Implements a 5-stage scoring pipeline:
    1. Normalize all metrics across frames (eye openness, motion, expression, pose, sharpness)
    2. Calculate quality penalties based on sharpness percentiles
    3. Calculate stability boosts based on local temporal variance
    4. Compute weighted composite scores from normalized metrics
    5. Apply temporal context window (sliding average)
    6. Final score = context_score × quality_penalty × stability_boost

    Attributes:
        config: RankingConfig with scoring weights and parameters
        quality_gate: QualityGate for calculating quality penalties
        stability_analyzer: StabilityAnalyzer for temporal stability boosts

    Example:
        >>> from sceneflow.shared.config import RankingConfig
        >>> config = RankingConfig()
        >>> scorer = FrameScorer(config)
        >>> scores = scorer.compute_scores(features)
        >>> best_score = max(scores, key=lambda s: s.final_score)
    """

    def __init__(self, config: RankingConfig):
        """
        Initialize the scorer with configuration.

        Args:
            config: RankingConfig instance with weights and parameters
        """
        self.config = config
        self.quality_gate = QualityGate(config.quality_gate_percentile)
        self.stability_analyzer = StabilityAnalyzer(config.local_stability_window)

        logger.debug(
            "Initialized FrameScorer: quality_gate=%.1f%%, stability_window=%d, context_window=%d",
            config.quality_gate_percentile,
            config.local_stability_window,
            config.context_window_size
        )

    def compute_scores(self, features: List[FrameFeatures]) -> List[FrameScore]:
        """
        Compute scores for all frames using multi-stage pipeline.

        Pipeline stages:
        1. Normalize all metrics across all frames
        2. Calculate quality penalties and stability boosts
        3. Compute weighted composite scores
        4. Apply multi-frame context window (temporal smoothing)
        5. Calculate final scores: context × quality_penalty × stability_boost

        Args:
            features: List of extracted FrameFeatures

        Returns:
            List of FrameScore with computed scores for each frame

        Note:
            Empty input returns empty list
        """
        if not features:
            logger.warning("No features provided to scorer")
            return []

        logger.info("Computing scores for %d frames", len(features))

        # Stage 1: Normalize metrics
        logger.debug("Stage 1: Normalizing metrics")

        # Eye openness: Use Gaussian normalization to favor normal/median values
        # (not too wide, not blinking)
        eye_openness_scores = MetricNormalizer.normalize_gaussian(
            [f.eye_openness for f in features],
            target=None,   # Use median as target
            sigma=None,    # Use data std
            inverse=False  # Peak at median (normal eye openness) gets score 1.0
        )

        # Motion: Lower is better (inverse normalization)
        motion_stability_scores = MetricNormalizer.normalize(
            [f.motion_magnitude for f in features],
            inverse=True
        )

        # Expression: Lower activity is better (inverse normalization)
        expression_neutrality_scores = MetricNormalizer.normalize(
            [f.expression_activity for f in features],
            inverse=True
        )

        # Pose: Lower deviation is better (inverse normalization)
        pose_stability_scores = MetricNormalizer.normalize(
            [f.pose_deviation for f in features],
            inverse=True
        )

        # Sharpness: Higher is better (normal normalization)
        visual_sharpness_scores = MetricNormalizer.normalize(
            [f.sharpness for f in features],
            inverse=False
        )

        # Stage 2: Calculate quality penalties and stability boosts
        logger.debug("Stage 2: Computing quality penalties and stability boosts")
        quality_penalties = self.quality_gate.calculate_penalties(features)
        stability_boosts = self.stability_analyzer.calculate_stability_boosts(features)

        # Stage 3: Compute composite scores (before context window)
        logger.debug("Stage 3: Computing weighted composite scores")
        frame_scores = []
        for i, feat in enumerate(features):
            composite = (
                self.config.eye_openness_weight * eye_openness_scores[i] +
                self.config.motion_stability_weight * motion_stability_scores[i] +
                self.config.expression_neutrality_weight * expression_neutrality_scores[i] +
                self.config.pose_stability_weight * pose_stability_scores[i] +
                self.config.visual_sharpness_weight * visual_sharpness_scores[i]
            )

            frame_scores.append(FrameScore(
                frame_index=feat.frame_index,
                timestamp=feat.timestamp,
                composite_score=composite,
                context_score=0.0,
                eye_openness_score=eye_openness_scores[i],
                motion_stability_score=motion_stability_scores[i],
                expression_neutrality_score=expression_neutrality_scores[i],
                pose_stability_score=pose_stability_scores[i],
                visual_sharpness_score=visual_sharpness_scores[i],
                quality_penalty=quality_penalties[i],
                stability_boost=stability_boosts[i],
                final_score=0.0  # Will be calculated after context window
            ))

        # Stage 4: Apply context window
        logger.debug("Stage 4: Applying temporal context window")
        context_scores = self._apply_context_window(frame_scores)

        # Stage 5: Calculate final scores using context-aware scores
        logger.debug("Stage 5: Computing final scores")
        for score, context in zip(frame_scores, context_scores):
            score.context_score = context
            # Use context score (temporal smoothing) instead of raw composite
            score.final_score = context * score.quality_penalty * score.stability_boost

        # Log statistics
        final_scores = [s.final_score for s in frame_scores]
        logger.info(
            "Scoring complete: min=%.4f, max=%.4f, mean=%.4f",
            min(final_scores),
            max(final_scores),
            np.mean(final_scores)
        )

        return frame_scores

    def _apply_context_window(self, frame_scores: List[FrameScore]) -> List[float]:
        """
        Apply sliding window averaging to composite scores.

        This rewards frames that are part of stable sequences by averaging
        their composite scores with neighboring frames within the window.

        Args:
            frame_scores: List of FrameScore objects with composite scores

        Returns:
            List of context scores (temporally smoothed composite scores)

        Note:
            Window is centered on each frame, with partial windows at boundaries
        """
        window_size = self.config.context_window_size
        half_window = window_size // 2

        composite_scores = [fs.composite_score for fs in frame_scores]
        context_scores = []

        for i in range(len(composite_scores)):
            # Calculate window boundaries
            start_idx = max(0, i - half_window)
            end_idx = min(len(composite_scores), i + half_window + 1)

            # Average scores within window
            window_values = composite_scores[start_idx:end_idx]
            context_score = np.mean(window_values)

            context_scores.append(float(context_score))

        return context_scores
