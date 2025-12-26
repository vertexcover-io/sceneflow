"""Frame ranking for podcast/talking head cut point detection."""

import asyncio
import logging
from typing import List, Optional

import cv2

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.extraction import FeatureExtractor
from sceneflow.shared.models import FrameFeatures, FrameScore, RankedFrame, RankingResult
from sceneflow.core.scorer import FrameScorer
from sceneflow.utils.video import VideoCapture, get_video_properties

logger = logging.getLogger(__name__)


class CutPointRanker:
    """
    Ranker for podcast/talking head videos.

    Finds frames where:
    - Eyes are open (not blinking)
    - Mouth is closed (not talking)
    """

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.config.validate()

        self.extractor = FeatureExtractor(min_face_confidence=self.config.min_face_confidence)
        self.scorer = FrameScorer(self.config)

        # Store last analysis internals for debugging/access
        self.last_features: Optional[List[FrameFeatures]] = None
        self.last_scores: Optional[List[FrameScore]] = None

        logger.debug(
            "Initialized CutPointRanker: eye_weight=%.2f, mouth_weight=%.2f",
            self.config.eye_weight,
            self.config.mouth_weight,
        )

    def rank_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = 1,
    ) -> RankingResult:
        features = self._extract_features(video_path, start_time, end_time, sample_rate)

        if not features:
            raise NoValidFramesError(
                f"No features extracted from {video_path} in range {start_time:.4f}-{end_time:.4f}s"
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

        self.last_features = features
        self.last_scores = scores

        return RankingResult(
            ranked_frames=ranked_frames,
            features=features,
            scores=scores,
        )

    def get_detailed_scores(
        self, video_path: str, start_time: float, end_time: float, sample_rate: int = 1
    ) -> List[FrameScore]:
        """Get detailed scoring breakdown for all frames."""
        features = self._extract_features(video_path, start_time, end_time, sample_rate)
        if not features:
            raise NoValidFramesError(f"No features extracted from {video_path}")
        scores = self.scorer.compute_scores(features)
        return sorted(scores, key=lambda x: x.final_score, reverse=True)

    def _extract_features(
        self, video_path: str, start_time: float, end_time: float, sample_rate: int
    ) -> List[FrameFeatures]:
        """Extract all face metrics from frames in the time range."""
        props = get_video_properties(video_path)
        fps = props.fps

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        with VideoCapture(video_path) as cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            features: List[FrameFeatures] = []
            current_frame_idx = start_frame

            while current_frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame %d", current_frame_idx)
                    break

                if (current_frame_idx - start_frame) % sample_rate == 0:
                    timestamp = current_frame_idx / fps

                    # Extract all face metrics
                    metrics = self.extractor.extract_face_metrics(frame)

                    features.append(
                        FrameFeatures(
                            frame_index=current_frame_idx,
                            timestamp=timestamp,
                            eye_openness=metrics.ear,
                            mouth_openness=metrics.mar,
                            face_detected=metrics.detected,
                            sharpness=metrics.sharpness,
                            face_center=metrics.center,
                            expression_activity=metrics.mar,  # For backward compatibility
                        )
                    )

                current_frame_idx += 1

        logger.info("Extracted features from %d frames", len(features))
        return features

    async def rank_frames_async(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = 1,
    ) -> RankingResult:
        """
        Async version of rank_frames.

        Rank frames in the given time range for optimal cut points.

        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_rate: Process every Nth frame (1 = all frames)

        Returns:
            RankingResult containing ranked_frames, features, and scores.
            Also stores in last_features and last_scores for backward compatibility.
        """
        features = await asyncio.to_thread(
            self._extract_features, video_path, start_time, end_time, sample_rate
        )

        if not features:
            raise NoValidFramesError(
                f"No features extracted from {video_path} in range {start_time:.4f}-{end_time:.4f}s"
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

        self.last_features = features
        self.last_scores = scores

        return RankingResult(
            ranked_frames=ranked_frames,
            features=features,
            scores=scores,
        )
