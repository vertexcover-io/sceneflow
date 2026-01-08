"""Frame ranking for podcast/talking head cut point detection."""

import logging
from typing import List, Optional

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.extraction import FeatureExtractor
from sceneflow.shared.models import FrameFeatures, FrameScore, RankedFrame, RankingResult
from sceneflow.core.scorer import FrameScorer

from sceneflow.utils.video import VideoSession

logger = logging.getLogger(__name__)


class CutPointRanker:
    """Ranker for podcast/talking head videos."""

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.config.validate()

        self.extractor = FeatureExtractor(min_face_confidence=self.config.min_face_confidence)
        self.scorer = FrameScorer(self.config)

        logger.debug(
            "Initialized CutPointRanker: eye_openness_weight=%.2f, expression_neutrality_weight=%.2f",
            self.config.eye_openness_weight,
            self.config.expression_neutrality_weight,
        )

    def rank_frames(
        self,
        session: "VideoSession",
        start_time: float,
        end_time: float,
        sample_rate: int = 1,
    ) -> RankingResult:
        """Rank frames in the given time range for optimal cut points.

        Args:
            session: VideoSession with open video
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_rate: Process every Nth frame (1 = all frames)
        """
        features = self._extract_features(session, start_time, end_time, sample_rate)

        if not features:
            raise NoValidFramesError(
                f"No features extracted from {session.video_path} in range {start_time:.4f}-{end_time:.4f}s"
            )

        logger.info("Extracted features from %d frames", len(features))

        scores = self.scorer.compute_scores(features)
        sorted_scores = sorted(scores, key=lambda x: x.final_score, reverse=True)

        ranked_frames = [
            RankedFrame(
                rank=i + 1,
                frame_index=score.frame_index,
                timestamp=score.timestamp,
                score=score.final_score,
            )
            for i, score in enumerate(sorted_scores)
        ]

        return RankingResult(ranked_frames=ranked_frames, features=features, scores=scores)

    def get_detailed_scores(
        self,
        session: "VideoSession",
        start_time: float,
        end_time: float,
        sample_rate: int = 1,
    ) -> List[FrameScore]:
        """Get detailed scoring breakdown for all frames."""
        features = self._extract_features(session, start_time, end_time, sample_rate)
        if not features:
            raise NoValidFramesError(f"No features extracted from {session.video_path}")
        scores = self.scorer.compute_scores(features)
        return sorted(scores, key=lambda x: x.final_score, reverse=True)

    def _extract_features(
        self,
        session: "VideoSession",
        start_time: float,
        end_time: float,
        sample_rate: int,
    ) -> List[FrameFeatures]:
        """Extract all face metrics from frames in the time range."""
        fps = session.properties.fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        features: List[FrameFeatures] = []

        for frame_idx, frame in session.iterate_frames(start_frame, end_frame, sample_rate):
            timestamp = frame_idx / fps
            metrics = self.extractor.extract_face_metrics(frame)

            features.append(
                FrameFeatures(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    eye_openness=metrics.ear,
                    mouth_openness=metrics.mar,
                    face_detected=metrics.detected,
                    sharpness=metrics.sharpness,
                    face_center=metrics.center,
                    expression_activity=metrics.mar,
                )
            )

        logger.info("Extracted features from %d frames", len(features))
        return features

    async def rank_frames_async(
        self,
        session: "VideoSession",
        start_time: float,
        end_time: float,
        sample_rate: int = 1,
    ) -> RankingResult:
        """Async version of rank_frames."""
        features = await self._extract_features_async(session, start_time, end_time, sample_rate)

        if not features:
            raise NoValidFramesError(
                f"No features extracted from {session.video_path} in range {start_time:.4f}-{end_time:.4f}s"
            )

        logger.info("Extracted features from %d frames", len(features))

        scores = self.scorer.compute_scores(features)
        sorted_scores = sorted(scores, key=lambda x: x.final_score, reverse=True)

        ranked_frames = [
            RankedFrame(
                rank=i + 1,
                frame_index=score.frame_index,
                timestamp=score.timestamp,
                score=score.final_score,
            )
            for i, score in enumerate(sorted_scores)
        ]

        return RankingResult(ranked_frames=ranked_frames, features=features, scores=scores)

    async def _extract_features_async(
        self,
        session: "VideoSession",
        start_time: float,
        end_time: float,
        sample_rate: int,
    ) -> List[FrameFeatures]:
        """Async version of _extract_features."""
        fps = session.properties.fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        features: List[FrameFeatures] = []

        async for frame_idx, frame in session.iterate_frames_async(
            start_frame, end_frame, sample_rate
        ):
            timestamp = frame_idx / fps
            metrics = self.extractor.extract_face_metrics(frame)

            features.append(
                FrameFeatures(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    eye_openness=metrics.ear,
                    mouth_openness=metrics.mar,
                    face_detected=metrics.detected,
                    sharpness=metrics.sharpness,
                    face_center=metrics.center,
                    expression_activity=metrics.mar,
                )
            )

        logger.info("Extracted features from %d frames", len(features))
        return features
