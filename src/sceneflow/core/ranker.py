"""Frame ranking for podcast/talking head cut point detection."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
            self._save_scores_txt(video_path, ranked_frames, features, scores)

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
        """Save annotated frames with landmarks color-coded."""
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

                # Annotate frame with landmarks
                annotated_frame = self._draw_landmarks(frame)

                output_filename = (
                    f"rank_{ranked_frame.rank:03d}_"
                    f"frame_{ranked_frame.frame_index}_"
                    f"timestamp_{ranked_frame.timestamp:.2f}.jpg"
                )
                cv2.imwrite(str(output_dir / output_filename), annotated_frame)

        logger.info("Saved frames to: %s", output_dir)

    def _save_scores_txt(
        self,
        video_path: str,
        ranked_frames: List[RankedFrame],
        features: List[FrameFeatures],
        scores: List[FrameScore]
    ) -> None:
        """Save detailed scores for all frames to scores.txt file."""
        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create lookup dictionaries
        frame_to_features = {f.frame_index: f for f in features}
        frame_to_scores = {s.frame_index: s for s in scores}
        frame_to_rank = {rf.frame_index: rf.rank for rf in ranked_frames}

        scores_file = output_dir / "scores.txt"

        with open(scores_file, 'w') as f:
            # Write header
            f.write("=" * 100 + "\n")
            f.write("DETAILED FRAME SCORES\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Video: {Path(video_path).name}\n")
            f.write(f"Total frames analyzed: {len(ranked_frames)}\n")
            f.write(f"Best cut point: Frame {ranked_frames[0].frame_index} at {ranked_frames[0].timestamp:.4f}s (score: {ranked_frames[0].score:.4f})\n")
            f.write("\n" + "=" * 100 + "\n\n")

            # Write column headers
            f.write(f"{'Rank':<6} {'Frame':<8} {'Time(s)':<10} {'Final':<8} {'Eye':<8} {'Mouth':<8} {'Sharp':<8} {'Consist':<8} {'EAR':<8} {'MAR':<8} {'Face':<6}\n")
            f.write("-" * 100 + "\n")

            # Write data for each frame in rank order
            for ranked_frame in ranked_frames:
                feature = frame_to_features.get(ranked_frame.frame_index)
                score = frame_to_scores.get(ranked_frame.frame_index)

                if not feature or not score:
                    continue

                # Format values
                rank_str = f"{ranked_frame.rank}"
                frame_str = f"{ranked_frame.frame_index}"
                time_str = f"{ranked_frame.timestamp:.4f}"
                final_str = f"{score.final_score:.4f}"
                eye_str = f"{score.eye_score:.4f}"
                mouth_str = f"{score.mouth_score:.4f}"
                sharp_str = f"{score.visual_sharpness_score:.4f}"
                consist_str = f"{score.motion_stability_score:.4f}"
                ear_str = f"{feature.eye_openness:.4f}"
                mar_str = f"{feature.mouth_openness:.4f}"
                face_str = "Yes" if feature.face_detected else "No"

                f.write(f"{rank_str:<6} {frame_str:<8} {time_str:<10} {final_str:<8} {eye_str:<8} {mouth_str:<8} {sharp_str:<8} {consist_str:<8} {ear_str:<8} {mar_str:<8} {face_str:<6}\n")

            # Write footer with legend
            f.write("\n" + "=" * 100 + "\n")
            f.write("LEGEND:\n")
            f.write("  Final    = Final weighted score (higher is better)\n")
            f.write("  Eye      = Eye openness score (1.0 = eyes fully open, 0.0 = blinking/squinting)\n")
            f.write("  Mouth    = Mouth closed score (1.0 = closed, 0.0 = talking/open)\n")
            f.write("  Sharp    = Sharpness score (1.0 = sharpest, 0.0 = blurriest)\n")
            f.write("  Consist  = Consistency score (1.0 = stable, 0.0 = sudden movement)\n")
            f.write("  EAR      = Eye Aspect Ratio (raw value, 0.25-0.32 = normal)\n")
            f.write("  MAR      = Mouth Aspect Ratio (raw value, 0.20-0.35 = closed)\n")
            f.write("  Face     = Whether a face was detected in the frame\n")
            f.write("=" * 100 + "\n")

        logger.info("Saved detailed scores to: %s", scores_file)

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

        if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
            return annotated

        landmarks = face.landmark_2d_106.astype(int)

        # Define color coding (BGR format for OpenCV)
        BLUE = (255, 0, 0)      # Eyes
        RED = (0, 0, 255)       # Mouth
        BEIGE = (220, 245, 245) # Other landmarks

        # Draw all 106 landmarks with appropriate colors
        for i, (x, y) in enumerate(landmarks):
            # Determine color based on index
            if INSIGHTFACE.LEFT_EYE_START <= i < INSIGHTFACE.LEFT_EYE_END or \
               INSIGHTFACE.RIGHT_EYE_START <= i < INSIGHTFACE.RIGHT_EYE_END:
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
