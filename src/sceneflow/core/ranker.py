"""Frame ranking for podcast/talking head cut point detection."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.extraction import FeatureExtractor
from sceneflow.shared.models import FrameFeatures, FrameScore, RankedFrame
from sceneflow.core.scorer import FrameScorer
from sceneflow.utils.video import VideoCapture, get_video_properties, cut_video
from sceneflow.shared.constants import INSIGHTFACE

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
        save_frames: bool = False,
        output_path: Optional[str] = None,
        save_logs: bool = False,
        vad_timestamps: Optional[List[Dict[str, float]]] = None,
    ) -> List[RankedFrame]:
        """
        Rank frames in the given time range for optimal cut points.

        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_rate: Process every Nth frame (1 = all frames)
            save_frames: If True, save annotated frames
            output_path: If provided, saves cut video to this path
            save_logs: If True, save detailed logs
            vad_timestamps: Optional list of VAD speech segments to include in logs

        Returns:
            List of RankedFrame sorted by score (best first).
            Access last_features and last_scores attributes for internals.
        """
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
                score=score.final_score,
            )
            for i, score in enumerate(sorted_scores)
        ]

        # Optional outputs
        if save_frames:
            self._save_ranked_frames(video_path, ranked_frames)
            # self._save_scores_txt(video_path, ranked_frames, features, scores)

        if save_logs:
            self._save_frame_logs(video_path, ranked_frames, features, scores, vad_timestamps)

        if output_path and ranked_frames:
            cut_video(video_path, ranked_frames[0].timestamp, output_path)

        # Store internals for later access
        self.last_features = features
        self.last_scores = scores

        return ranked_frames

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

    def _save_ranked_frames(self, video_path: str, ranked_frames: List[RankedFrame]) -> None:
        """Save annotated frames with landmarks color-coded."""
        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Tuple[path, annotated frame images]
        frames_to_write: List[Tuple[str, np.ndarray]] = []

        with VideoCapture(video_path) as cap:
            for ranked_frame in ranked_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ranked_frame.frame_index)
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Failed to read frame %d for saving", ranked_frame.frame_index)
                    continue

                # Annotate frame with landmarks
                annotated_frame = self._draw_landmarks(frame)

                output_filename = (
                    f"rank_{ranked_frame.rank:03d}_"
                    f"frame_{ranked_frame.frame_index}_"
                    f"timestamp_{ranked_frame.timestamp:.2f}.jpg"
                )
                output_path = str(output_dir / output_filename)
                frames_to_write.append((output_path, annotated_frame))

        if not frames_to_write:
            logger.warning("No frames to save")
            return

        max_workers = min(8, os.cpu_count() or 4, len(frames_to_write))
        saved_count = 0
        failed_count = 0

        def write_frame(args: Tuple[str, np.ndarray]) -> bool:
            path, frame = args
            try:
                cv2.imwrite(path, frame)
                return True
            except Exception as e:
                logger.error("Failed to write frame to %s: %s", path, e)
                return False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(write_frame, item) for item in frames_to_write]

            for future in as_completed(futures):
                if future.result():
                    saved_count += 1
                else:
                    failed_count += 1

        if failed_count > 0:
            logger.warning(
                "Saved %d frames, %d failed to write to: %s", saved_count, failed_count, output_dir
            )
        else:
            logger.info("Saved %d annotated frames to: %s", saved_count, output_dir)

    def _draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw InsightFace 106 landmarks on frame with color coding.

        Colors:
        - Blue: Eyes (indices 37-48)
        - Red: Mouth (indices 52-71)
        - Beige: Other landmarks

        Args:
            frame: Input frame (BGR format)

        Returns:
            Frame with landmarks drawn
        """
        # Make a copy to avoid modifying original
        annotated = frame.copy()

        # Get face with landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.extractor.app.get(rgb_frame)

        if not faces:
            return annotated

        # Use the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        if not hasattr(face, "landmark_2d_106") or face.landmark_2d_106 is None:
            return annotated

        landmarks = face.landmark_2d_106.astype(int)

        # Define color coding (BGR format for OpenCV)
        BLUE = (255, 0, 0)  # Eyes
        RED = (0, 0, 255)  # Mouth
        BEIGE = (220, 245, 245)  # Other landmarks

        # Draw all 106 landmarks with appropriate colors
        for i, (x, y) in enumerate(landmarks):
            # Determine color based on index
            if (
                INSIGHTFACE.LEFT_EYE_START <= i < INSIGHTFACE.LEFT_EYE_END
                or INSIGHTFACE.RIGHT_EYE_START <= i < INSIGHTFACE.RIGHT_EYE_END
            ):
                color = BLUE
            elif INSIGHTFACE.MOUTH_OUTER_START <= i < INSIGHTFACE.MOUTH_OUTER_END:
                color = RED
            else:
                color = BEIGE

            cv2.circle(annotated, (x, y), 2, color, -1)

        return annotated

    def _save_frame_logs(
        self,
        video_path: str,
        ranked_frames: List[RankedFrame],
        features: List[FrameFeatures],
        scores: List[FrameScore],
        vad_timestamps: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """Save detailed analysis logs."""
        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_to_features = {f.frame_index: f for f in features}
        frame_to_scores = {s.frame_index: s for s in scores}

        # Tuple[path, frame logs to write]
        logs_to_write: List[Tuple[str, dict]] = []

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

            output_filename = f"rank_{ranked_frame.rank:03d}_frame_{ranked_frame.frame_index}.json"
            output_path = str(output_dir / output_filename)
            logs_to_write.append((output_path, log_data))

        # Add VAD timestamps log if available
        if vad_timestamps:
            vad_log_data = {
                "speech_segments": [{"start": seg.start, "end": seg.end} for seg in vad_timestamps],
                "total_segments": len(vad_timestamps),
                "speech_end_time": vad_timestamps[-1].end if vad_timestamps else 0.0,
            }
            logs_to_write.append((str(output_dir / "vad_timestamps.json"), vad_log_data))

        if not logs_to_write:
            logger.warning("No logs to save")
            return

        max_workers = min(8, os.cpu_count() or 4, len(logs_to_write))
        saved_count = 0
        failed_count = 0

        def write_json(args: Tuple[str, dict]) -> bool:
            path, data = args
            try:
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                return True
            except Exception as e:
                logger.error("Failed to write log to %s: %s", path, e)
                return False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(write_json, item) for item in logs_to_write]

            for future in as_completed(futures):
                if future.result():
                    saved_count += 1
                else:
                    failed_count += 1

        if vad_timestamps:
            logger.info("Saved VAD timestamps with %d segments", len(vad_timestamps))

        if failed_count > 0:
            logger.warning(
                "Saved %d logs, %d failed to write to: %s", saved_count, failed_count, output_dir
            )
        else:
            logger.info("Saved %d logs to: %s", saved_count, output_dir)
