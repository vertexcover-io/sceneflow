"""Output utilities for saving analysis results."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sceneflow.shared.models import FrameFeatures, FrameScore, RankedFrame
from sceneflow.utils.video import VideoCapture
from sceneflow.shared.constants import INSIGHTFACE
from sceneflow.extraction import FeatureExtractor

logger = logging.getLogger(__name__)


def save_annotated_frames(
    video_path: str,
    ranked_frames: List[RankedFrame],
    extractor: FeatureExtractor,
) -> None:
    """Save annotated frames with landmarks color-coded.

    Args:
        video_path: Path to the video file
        ranked_frames: List of ranked frames to save
        extractor: FeatureExtractor instance for drawing landmarks
    """
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
            annotated_frame = _draw_landmarks(frame, extractor)

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


def save_analysis_logs(
    video_path: str,
    ranked_frames: List[RankedFrame],
    features: List[FrameFeatures],
    scores: List[FrameScore],
    vad_timestamps: Optional[List[Dict[str, float]]] = None,
) -> None:
    """Save detailed analysis logs.

    Args:
        video_path: Path to the video file
        ranked_frames: List of ranked frames
        features: List of extracted features
        scores: List of computed scores
        vad_timestamps: Optional list of VAD speech segments
    """
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


def _draw_landmarks(frame: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
    """Draw InsightFace 106 landmarks on frame with color coding.

    Colors:
    - Blue: Eyes (indices 37-48)
    - Red: Mouth (indices 52-71)
    - Beige: Other landmarks

    Args:
        frame: Input frame (BGR format)
        extractor: FeatureExtractor instance for accessing InsightFace app

    Returns:
        Frame with landmarks drawn
    """
    # Make a copy to avoid modifying original
    annotated = frame.copy()

    # Get face with landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = extractor.app.get(rgb_frame)

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
