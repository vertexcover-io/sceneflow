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
from sceneflow.shared.constants import INSIGHTFACE, MAX_FRAME_WORKERS, DEFAULT_CPU_COUNT_FALLBACK
from sceneflow.extraction import FeatureExtractor
from sceneflow.utils.video import VideoSession

logger = logging.getLogger(__name__)


def save_annotated_frames(
    session: "VideoSession",
    ranked_frames: List[RankedFrame],
    extractor: FeatureExtractor,
) -> None:
    """Save annotated frames with landmarks color-coded.

    Args:
        session: VideoSession with open video (uses cached frames when available)
        ranked_frames: List of ranked frames to save
        extractor: FeatureExtractor instance for drawing landmarks
    """
    video_base_name = Path(session.video_path).stem
    output_dir = Path("output") / video_base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_to_write: List[Tuple[str, np.ndarray]] = []

    for ranked_frame in ranked_frames:
        try:
            frame = session.get_frame(ranked_frame.frame_index)
            annotated_frame = _draw_landmarks(frame, extractor)

            output_filename = (
                f"rank_{ranked_frame.rank:03d}_"
                f"frame_{ranked_frame.frame_index}_"
                f"timestamp_{ranked_frame.timestamp:.2f}.jpg"
            )
            output_path = str(output_dir / output_filename)
            frames_to_write.append((output_path, annotated_frame))
        except Exception as e:
            logger.warning("Failed to read frame %d for saving: %s", ranked_frame.frame_index, e)

    if not frames_to_write:
        logger.warning("No frames to save")
        return

    max_workers = min(
        MAX_FRAME_WORKERS, os.cpu_count() or DEFAULT_CPU_COUNT_FALLBACK, len(frames_to_write)
    )
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
    """Save detailed analysis logs."""
    video_base_name = Path(video_path).stem
    output_dir = Path("output") / video_base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_to_features = {f.frame_index: f for f in features}
    frame_to_scores = {s.frame_index: s for s in scores}

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

    max_workers = min(
        MAX_FRAME_WORKERS, os.cpu_count() or DEFAULT_CPU_COUNT_FALLBACK, len(logs_to_write)
    )
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
    """Draw InsightFace 106 landmarks on frame with color coding."""
    annotated = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = extractor.app.get(rgb_frame)

    if not faces:
        return annotated

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    if not hasattr(face, "landmark_2d_106") or face.landmark_2d_106 is None:
        return annotated

    landmarks = face.landmark_2d_106.astype(int)

    BLUE = (255, 0, 0)
    RED = (0, 0, 255)
    BEIGE = (220, 245, 245)

    for i, (x, y) in enumerate(landmarks):
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
