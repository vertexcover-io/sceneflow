"""Frame ranking for podcast/talking head cut point detection."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.extraction import FeatureExtractor
from sceneflow.shared.models import FrameFeatures, FrameScore, RankedFrame
from sceneflow.core.scorer import FrameScorer
from sceneflow.utils.video import VideoCapture, get_video_properties, cut_video

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

        self.extractor = FeatureExtractor(
            min_face_confidence=self.config.min_face_confidence
        )
        self.scorer = FrameScorer(self.config)

        logger.debug(
            "Initialized CutPointRanker: eye_weight=%.2f, mouth_weight=%.2f",
            self.config.eye_weight,
            self.config.mouth_weight
        )

    def rank_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = 1,
        save_frames: bool = False,
        save_video: bool = False,
        output_path: Optional[str] = None,
        save_logs: bool = False,
        vad_timestamps: Optional[List[Dict[str, float]]] = None,
        return_internals: bool = False
    ) -> Union[List[RankedFrame], Tuple[List[RankedFrame], List[FrameFeatures], List[FrameScore]]]:
        """
        Rank frames in the given time range for optimal cut points.

        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_rate: Process every Nth frame (1 = all frames)
            save_frames: If True, save annotated frames
            save_video: If True, cut video from start to best timestamp
            output_path: Optional custom path for saved video
            save_logs: If True, save detailed logs
            vad_timestamps: Optional list of VAD speech segments to include in logs
            return_internals: If True, return (frames, features, scores) tuple

        Returns:
            List of RankedFrame sorted by score (best first), or tuple with internals
        """
        logger.info(
            "Ranking frames in range %.4f-%.4fs (sample_rate=%d)",
            start_time, end_time, sample_rate
        )

        # Extract features from all frames
        features = self._extract_features(video_path, start_time, end_time, sample_rate)

        if not features:
            raise NoValidFramesError(
                f"No features extracted from {video_path} in range {start_time:.4f}-{end_time:.4f}s"
            )

        logger.info("Extracted features from %d frames", len(features))

        # Compute scores
        scores = self.scorer.compute_scores(features)

        # Sort by final score (descending)
        sorted_scores = sorted(scores, key=lambda x: x.final_score, reverse=True)

        # Create ranked frame objects
        ranked_frames = [
            RankedFrame(
                rank=i + 1,
                frame_index=score.frame_index,
                timestamp=score.timestamp,
                score=score.final_score
            )
            for i, score in enumerate(sorted_scores)
        ]

        logger.info(
            "Best cut point: %.4fs (score: %.3f)",
            ranked_frames[0].timestamp,
            ranked_frames[0].score
        )

        # Optional outputs
        if save_frames:
            self._save_ranked_frames(video_path, ranked_frames)

        if save_logs:
            self._save_frame_logs(video_path, ranked_frames, features, scores, vad_timestamps)

        if save_video and ranked_frames:
            self._save_cut_video(video_path, ranked_frames[0].timestamp, output_path=output_path)

        if return_internals:
            return ranked_frames, features, scores
        return ranked_frames

    def get_detailed_scores(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = 1
    ) -> List[FrameScore]:
        """Get detailed scoring breakdown for all frames."""
        features = self._extract_features(video_path, start_time, end_time, sample_rate)
        if not features:
            raise NoValidFramesError(f"No features extracted from {video_path}")
        scores = self.scorer.compute_scores(features)
        return sorted(scores, key=lambda x: x.final_score, reverse=True)

    def _extract_features(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int
    ) -> List[FrameFeatures]:
        """Extract all face metrics from frames in the time range."""
        props = get_video_properties(video_path)
        fps = props.fps

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        logger.info(
            "Extracting features from frames %d to %d",
            start_frame, end_frame
        )

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

                    features.append(FrameFeatures(
                        frame_index=current_frame_idx,
                        timestamp=timestamp,
                        eye_openness=metrics.ear,
                        mouth_openness=metrics.mar,
                        face_detected=metrics.detected,
                        sharpness=metrics.sharpness,
                        face_center=metrics.center,
                        expression_activity=metrics.mar,  # For backward compatibility
                    ))

                current_frame_idx += 1

        logger.info("Extracted features from %d frames", len(features))
        return features

    def _save_ranked_frames(self, video_path: str, ranked_frames: List[RankedFrame]) -> None:
        """Save annotated frames."""
        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving %d frames to: %s", len(ranked_frames), output_dir)

        with VideoCapture(video_path) as cap:
            for ranked_frame in ranked_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ranked_frame.frame_index)
                ret, frame = cap.read()

                if not ret:
                    continue

                output_filename = (
                    f"rank_{ranked_frame.rank:03d}_"
                    f"frame_{ranked_frame.frame_index}_"
                    f"timestamp_{ranked_frame.timestamp:.2f}.jpg"
                )
                cv2.imwrite(str(output_dir / output_filename), frame)

        logger.info("Saved frames to: %s", output_dir)

    def _save_frame_logs(
        self,
        video_path: str,
        ranked_frames: List[RankedFrame],
        features: List[FrameFeatures],
        scores: List[FrameScore],
        vad_timestamps: Optional[List[Dict[str, float]]] = None
    ) -> None:
        """Save detailed analysis logs."""
        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_to_features = {f.frame_index: f for f in features}
        frame_to_scores = {s.frame_index: s for s in scores}

        for ranked_frame in ranked_frames:
            feature = frame_to_features.get(ranked_frame.frame_index)
            score = frame_to_scores.get(ranked_frame.frame_index)

            if not feature or not score:
                continue

            log_data = {
                "rank": ranked_frame.rank,
                "frame_index": ranked_frame.frame_index,
                "timestamp": ranked_frame.timestamp,
                "final_score": ranked_frame.score,
                "features": {
                    "eye_openness": feature.eye_openness,
                    "mouth_openness": feature.mouth_openness,
                    "sharpness": feature.sharpness,
                    "face_center": feature.face_center,
                },
                "scores": {
                    "eye": score.eye_score,
                    "mouth": score.mouth_score,
                    "sharpness": score.visual_sharpness_score,
                    "consistency": score.motion_stability_score,
                },
            }

            output_filename = (
                f"rank_{ranked_frame.rank:03d}_"
                f"frame_{ranked_frame.frame_index}.json"
            )
            with open(output_dir / output_filename, 'w') as f:
                json.dump(log_data, f, indent=2)

        if vad_timestamps:
            vad_log_data = {
                "speech_segments": vad_timestamps,
                "total_segments": len(vad_timestamps),
                "speech_end_time": vad_timestamps[-1]["end"] if vad_timestamps else 0.0
            }
            with open(output_dir / "vad_timestamps.json", 'w') as f:
                json.dump(vad_log_data, f, indent=2)
            logger.info("Saved VAD timestamps with %d segments", len(vad_timestamps))

        logger.info("Saved logs to: %s", output_dir)

    def _save_cut_video(
        self,
        video_path: str,
        cut_timestamp: float,
        output_path: Optional[str] = None
    ) -> str:
        """
        Cut video from start to the specified timestamp using FFmpeg.

        Args:
            video_path: Path to input video file
            cut_timestamp: Timestamp where to cut the video (in seconds)
            output_path: Optional custom output path for the cut video.
                        If None, saves to output/<video_name>_cut.mp4

        Returns:
            Path to the saved cut video

        Raises:
            FFmpegNotFoundError: If ffmpeg is not installed
            FFmpegExecutionError: If ffmpeg command fails
        """
        return cut_video(video_path, cut_timestamp, output_path)
