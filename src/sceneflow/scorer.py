import numpy as np
from typing import List
from .models import FrameFeatures, FrameScore
from .normalizer import MetricNormalizer
from .config import RankingConfig
from .quality_gating import QualityGate
from .stability_analyzer import StabilityAnalyzer


class FrameScorer:
    def __init__(self, config: RankingConfig):
        self.config = config
        self.quality_gate = QualityGate(config.quality_gate_percentile)
        self.stability_analyzer = StabilityAnalyzer(config.local_stability_window)

    def compute_scores(self, features: List[FrameFeatures]) -> List[FrameScore]:
        """
        Multi-stage scoring pipeline:
        1. Normalize all metrics across all frames
        2. Calculate quality penalties and stability boosts
        3. Compute weighted composite scores
        4. Apply multi-frame context window (temporal smoothing)
        5. Calculate final scores: context * quality_penalty * stability_boost
        """
        if not features:
            return []

        eye_openness_scores = MetricNormalizer.normalize_gaussian(
            [f.eye_openness for f in features],
            target=0.32,
            sigma=0.05,
            inverse=False
        )

        motion_stability_scores = MetricNormalizer.normalize(
            [f.motion_magnitude for f in features],
            inverse=True
        )

        expression_neutrality_scores = MetricNormalizer.normalize(
            [f.expression_activity for f in features],
            inverse=True
        )

        pose_stability_scores = MetricNormalizer.normalize(
            [f.pose_deviation for f in features],
            inverse=True
        )

        visual_sharpness_scores = MetricNormalizer.normalize(
            [f.sharpness for f in features],
            inverse=False
        )

        # Stage 2: Calculate quality penalties and stability boosts
        quality_penalties = self.quality_gate.calculate_penalties(features)
        stability_boosts = self.stability_analyzer.calculate_stability_boosts(features)

        # Stage 3: Compute composite scores (before context window)
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

        context_scores = self._apply_context_window(frame_scores)

        for i, (score, context) in enumerate(zip(frame_scores, context_scores)):
            score.context_score = context
            score.final_score = context * score.quality_penalty * score.stability_boost

            recency_factor = 1.0 + (i / len(frame_scores)) * 0.05
            score.final_score *= recency_factor

        return frame_scores

    def _apply_context_window(self, frame_scores: List[FrameScore]) -> List[float]:
        """
        Apply sliding window averaging to composite scores.
        This rewards frames that are part of stable sequences.
        """
        window_size = self.config.context_window_size
        half_window = window_size // 2

        composite_scores = [fs.composite_score for fs in frame_scores]
        context_scores = []

        for i in range(len(composite_scores)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(composite_scores), i + half_window + 1)

            window_values = composite_scores[start_idx:end_idx]
            context_score = np.mean(window_values)

            context_scores.append(float(context_score))

        return context_scores
