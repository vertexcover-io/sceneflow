"""Frame ranking module for cut point detection.

This module provides the CutPointRanker class which orchestrates the complete
ranking pipeline: feature extraction, scoring, and selection of optimal cut points.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.extraction import FeatureExtractor, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_OUTER_INDICES
from sceneflow.shared.models import FrameFeatures, FrameScore, RankedFrame
from sceneflow.core.scorer import FrameScorer
from sceneflow.utils.video import VideoCapture, get_video_properties, cut_video

logger = logging.getLogger(__name__)


class CutPointRanker:
    """
    Orchestrates frame ranking for optimal cut point detection.

    This class coordinates feature extraction, scoring, and ranking to identify
    the best frames for cutting a video. It supports multi-stage ranking with
    temporal coherence and quality filtering.

    Pipeline:
        1. Extract features from frames (eye openness, motion, expression, etc.)
        2. Compute multi-factor scores with quality gating and stability boosts
        3. Rank frames by final score
        4. Optionally save annotated frames and cut video

    Attributes:
        config: RankingConfig instance with scoring weights and parameters
        extractor: FeatureExtractor instance for visual feature extraction
        scorer: FrameScorer instance for computing frame scores

    Example:
        >>> ranker = CutPointRanker()
        >>> ranked = ranker.rank_frames("video.mp4", start_time=5.0, end_time=10.0)
        >>> best_frame = ranked[0]
        >>> print(f"Best cut at {best_frame.timestamp:.2f}s")
    """

    def __init__(self, config: Optional[RankingConfig] = None):
        """
        Initialize the ranker with configuration.

        Args:
            config: Optional RankingConfig. If None, uses defaults.

        Raises:
            InvalidConfigError: If configuration is invalid
        """
        self.config = config or RankingConfig()
        self.config.validate()

        self.extractor = FeatureExtractor(
            center_weighting_strength=self.config.center_weighting_strength,
            min_face_confidence=self.config.min_face_confidence
        )
        self.scorer = FrameScorer(self.config)

        logger.debug(
            "Initialized CutPointRanker with weights: "
            "eye=%.2f, motion=%.2f, expression=%.2f, pose=%.2f, sharpness=%.2f",
            self.config.eye_openness_weight,
            self.config.motion_stability_weight,
            self.config.expression_neutrality_weight,
            self.config.pose_stability_weight,
            self.config.visual_sharpness_weight
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
        return_internals: bool = False
    ) -> Union[List[RankedFrame], Tuple[List[RankedFrame], List[FrameFeatures], List[FrameScore]]]:
        """
        Rank frames in the given time range for optimal cut points.

        Uses multi-stage ranking with temporal diversity filtering.

        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_rate: Process every Nth frame (1 = all frames)
            save_frames: If True, save frames with InsightFace 106 landmarks
            save_video: If True, cut video from start to best timestamp
            output_path: Optional custom path for saved video
            save_logs: If True, save detailed analysis data to JSONL files
            return_internals: If True, return (frames, features, scores) tuple

        Returns:
            If return_internals=False: List of RankedFrame sorted by score (best first)
            If return_internals=True: Tuple of (ranked_frames, features, scores)

        Raises:
            VideoOpenError: If video cannot be opened
            NoValidFramesError: If no valid frames found for ranking

        Example:
            >>> ranker = CutPointRanker()
            >>> ranked = ranker.rank_frames("video.mp4", 5.0, 10.0, sample_rate=2)
            >>> print(f"Found {len(ranked)} candidate frames")
        """
        logger.info(
            "Ranking frames in range %.2f-%.2fs (sample_rate=%d)",
            start_time, end_time, sample_rate
        )

        # Extract features from all frames
        features = self._extract_features(video_path, start_time, end_time, sample_rate)

        if not features:
            raise NoValidFramesError(
                f"No features extracted from {video_path} in range {start_time:.2f}-{end_time:.2f}s"
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
            "Ranking complete. Best frame: #%d at %.2fs (score: %.4f)",
            ranked_frames[0].frame_index,
            ranked_frames[0].timestamp,
            ranked_frames[0].score
        )

        # Optional outputs
        if save_frames:
            self._save_ranked_frames(video_path, ranked_frames)

        if save_logs:
            self._save_frame_logs(video_path, ranked_frames, features, scores)

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
        """
        Get detailed scoring breakdown for all frames.

        Useful for debugging and understanding the ranking.
        Includes quality penalties and stability boosts.

        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_rate: Process every Nth frame (1 = all frames)

        Returns:
            List of FrameScore sorted by final score (best first)

        Raises:
            VideoOpenError: If video cannot be opened
            NoValidFramesError: If no valid frames found

        Example:
            >>> ranker = CutPointRanker()
            >>> scores = ranker.get_detailed_scores("video.mp4", 5.0, 10.0)
            >>> for score in scores[:3]:
            ...     print(f"Frame {score.frame_index}: {score.final_score:.4f}")
        """
        features = self._extract_features(video_path, start_time, end_time, sample_rate)

        if not features:
            raise NoValidFramesError(f"No features extracted from {video_path}")

        scores = self.scorer.compute_scores(features)
        sorted_scores = sorted(scores, key=lambda x: x.final_score, reverse=True)
        return sorted_scores

    def _extract_features(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int
    ) -> List[FrameFeatures]:
        """
        Extract features from all frames in the time range.

        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_rate: Process every Nth frame

        Returns:
            List of FrameFeatures for all sampled frames

        Raises:
            VideoOpenError: If video cannot be opened
        """
        # Get video properties
        props = get_video_properties(video_path)
        fps = props.fps

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        logger.info(
            "Extracting features from frames %d to %d (%.2f-%.2fs, fps=%.2f)",
            start_frame, end_frame, start_time, end_time, fps
        )

        # Open video with context manager
        with VideoCapture(video_path) as cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Reset extractor state for fresh optical flow
            self.extractor.reset()

            features: List[FrameFeatures] = []
            current_frame_idx = start_frame

            while current_frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        "Failed to read frame %d, stopping extraction",
                        current_frame_idx
                    )
                    break

                # Process every Nth frame
                if (current_frame_idx - start_frame) % sample_rate == 0:
                    timestamp = current_frame_idx / fps

                    # Extract all faces and get aggregated metrics
                    num_faces, face_features_list, aggregated_metrics = \
                        self.extractor.extract_all_faces(frame)

                    # Extract frame-level metrics (not face-specific)
                    motion_magnitude = self.extractor.extract_motion_magnitude(frame)
                    sharpness = self.extractor.extract_visual_sharpness(frame)

                    features.append(FrameFeatures(
                        frame_index=current_frame_idx,
                        timestamp=timestamp,
                        eye_openness=aggregated_metrics.eye_openness,
                        motion_magnitude=motion_magnitude,
                        expression_activity=aggregated_metrics.expression_activity,
                        pose_deviation=aggregated_metrics.pose_deviation,
                        sharpness=sharpness,
                        num_faces=num_faces,
                        individual_faces=face_features_list if face_features_list else None
                    ))

                current_frame_idx += 1

        logger.info("Feature extraction complete: %d frames processed", len(features))
        return features

    def _save_ranked_frames(self, video_path: str, ranked_frames: List[RankedFrame]) -> None:
        """
        Save annotated frames with InsightFace 106 landmarks highlighted.

        Args:
            video_path: Path to video file
            ranked_frames: List of ranked frames to save

        Example output: output/video_name/rank_001_frame_1234_timestamp_5.67.jpg
        """
        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving %d annotated frames to: %s", len(ranked_frames), output_dir)

        with VideoCapture(video_path) as cap:
            for ranked_frame in ranked_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ranked_frame.frame_index)
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Failed to read frame %d for saving", ranked_frame.frame_index)
                    continue

                # Draw landmarks
                annotated_frame = self._draw_feature_boxes(frame.copy())

                # Save with descriptive filename
                output_filename = (
                    f"rank_{ranked_frame.rank:03d}_"
                    f"frame_{ranked_frame.frame_index}_"
                    f"timestamp_{ranked_frame.timestamp:.2f}.jpg"
                )
                output_path = output_dir / output_filename

                cv2.imwrite(str(output_path), annotated_frame)

        logger.info("Saved %d annotated frames to: %s", len(ranked_frames), output_dir)

    def _save_frame_logs(
        self,
        video_path: str,
        ranked_frames: List[RankedFrame],
        features: List[FrameFeatures],
        scores: List[FrameScore]
    ) -> None:
        """
        Save detailed analysis logs as JSONL files.

        Args:
            video_path: Path to video file
            ranked_frames: List of ranked frames
            features: List of frame features
            scores: List of frame scores
        """
        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving frame logs to: %s", output_dir)

        # Create lookup dictionaries
        frame_to_features = {f.frame_index: f for f in features}
        frame_to_scores = {s.frame_index: s for s in scores}

        for ranked_frame in ranked_frames:
            frame_idx = ranked_frame.frame_index
            feature = frame_to_features.get(frame_idx)
            score = frame_to_scores.get(frame_idx)

            if not feature or not score:
                logger.warning("Missing data for frame %d, skipping log", frame_idx)
                continue

            log_data = {
                "metadata": {
                    "rank": ranked_frame.rank,
                    "frame_index": ranked_frame.frame_index,
                    "timestamp": ranked_frame.timestamp
                },
                "raw_features": {
                    "eye_openness": feature.eye_openness,
                    "motion_magnitude": feature.motion_magnitude,
                    "expression_activity": feature.expression_activity,
                    "pose_deviation": feature.pose_deviation,
                    "sharpness": feature.sharpness,
                    "num_faces": feature.num_faces
                },
                "individual_faces": [],
                "normalized_scores": {
                    "eye_openness_score": score.eye_openness_score,
                    "motion_stability_score": score.motion_stability_score,
                    "expression_neutrality_score": score.expression_neutrality_score,
                    "pose_stability_score": score.pose_stability_score,
                    "visual_sharpness_score": score.visual_sharpness_score
                },
                "score_breakdown": {
                    "composite_score": score.composite_score,
                    "context_score": score.context_score,
                    "quality_penalty": score.quality_penalty,
                    "stability_boost": score.stability_boost,
                    "final_score": score.final_score
                },
                "configuration": {
                    "eye_openness_weight": self.config.eye_openness_weight,
                    "motion_stability_weight": self.config.motion_stability_weight,
                    "expression_neutrality_weight": self.config.expression_neutrality_weight,
                    "pose_stability_weight": self.config.pose_stability_weight,
                    "visual_sharpness_weight": self.config.visual_sharpness_weight,
                    "context_window_size": self.config.context_window_size,
                    "quality_gate_percentile": self.config.quality_gate_percentile,
                    "local_stability_window": self.config.local_stability_window,
                    "center_weighting_strength": self.config.center_weighting_strength,
                    "min_face_confidence": self.config.min_face_confidence
                }
            }

            # Add individual face data if available
            if feature.individual_faces:
                for face in feature.individual_faces:
                    log_data["individual_faces"].append({
                        "face_index": face.face_index,
                        "bbox": face.bbox,
                        "center_distance": face.center_distance,
                        "center_weight": face.center_weight,
                        "eye_openness": face.eye_openness,
                        "expression_activity": face.expression_activity,
                        "pose_deviation": face.pose_deviation
                    })

            # Save as JSONL
            output_filename = (
                f"rank_{ranked_frame.rank:03d}_"
                f"frame_{ranked_frame.frame_index}_"
                f"timestamp_{ranked_frame.timestamp:.2f}.jsonl"
            )
            output_path = output_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)

        logger.info("Saved %d log files to: %s", len(ranked_frames), output_dir)

    def _draw_feature_boxes(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw InsightFace 106 landmark dots for all detected faces.

        Landmarks are color-coded:
        - Eyes: Blue
        - Mouth: Red
        - Other: Beige

        Args:
            frame: Input frame (BGR format)

        Returns:
            Annotated frame with landmarks drawn
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.extractor.app.get(rgb_frame)

        if not faces:
            return frame

        # Draw features for ALL detected faces
        for face_idx, face in enumerate(faces):
            # Draw bounding box
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Add face number label
            label = f"Face {face_idx + 1}"
            cv2.putText(
                frame, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            # Check if 106 landmarks are available
            if (self.extractor.has_106_landmarks and
                hasattr(face, 'landmark_2d_106') and
                face.landmark_2d_106 is not None):

                landmarks = face.landmark_2d_106.astype(int)

                # Draw all landmarks with color coding (BGR format)
                for i, point in enumerate(landmarks):
                    # Determine color based on landmark group
                    if i in LEFT_EYE_INDICES or i in RIGHT_EYE_INDICES:
                        color = (255, 0, 0)  # Blue for eyes
                    elif i in MOUTH_OUTER_INDICES:
                        color = (0, 0, 255)  # Red for mouth
                    else:
                        color = (200, 160, 75)  # Beige for other landmarks

                    # Draw dot
                    cv2.circle(frame, tuple(point), 1, color, -1)
            else:
                # Fallback: draw 5-point landmarks if available
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int)
                    colors = [
                        (255, 0, 0),    # Left eye - Blue
                        (255, 0, 0),    # Right eye - Blue
                        (0, 255, 255),  # Nose - Yellow
                        (0, 0, 255),    # Left mouth - Red
                        (0, 0, 255)     # Right mouth - Red
                    ]

                    for point, color in zip(landmarks, colors):
                        cv2.circle(frame, (point[0], point[1]), 3, color, -1)

        # Add legend (only once for all faces)
        legend_y = 30
        cv2.putText(
            frame, f"Faces: {len(faces)}", (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        cv2.putText(
            frame, "Eyes (Blue)", (10, legend_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
        cv2.putText(
            frame, "Mouth (Red)", (10, legend_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        cv2.putText(
            frame, "Other (Beige)", (10, legend_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 160, 75), 1
        )

        return frame

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
