"""Output utilities for saving analysis results."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import cv2
import numpy as np

from sceneflow.shared.models import FrameFeatures, FrameScore, RankedFrame
from sceneflow.shared.constants import INSIGHTFACE
from sceneflow.extraction import FeatureExtractor
from sceneflow.utils.video import VideoSession

logger = logging.getLogger(__name__)


def _prepare_annotated_frames(
    session: "VideoSession",
    ranked_frames: List[RankedFrame],
    extractor: FeatureExtractor,
) -> Tuple[Path, List[Tuple[Path, bytes]]]:
    video_base_name = Path(session.video_path).stem
    output_dir = Path("output") / video_base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_to_write: List[Tuple[Path, bytes]] = []

    for ranked_frame in ranked_frames:
        try:
            frame = session.get_frame(ranked_frame.frame_index)
            annotated_frame = _draw_landmarks(frame, extractor)

            output_filename = (
                f"rank_{ranked_frame.rank:03d}_"
                f"frame_{ranked_frame.frame_index}_"
                f"timestamp_{ranked_frame.timestamp:.2f}.jpg"
            )
            output_path = output_dir / output_filename

            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frames_to_write.append((output_path, buffer.tobytes()))
        except Exception as e:
            logger.warning("Failed to prepare frame %d: %s", ranked_frame.frame_index, e)

    return output_dir, frames_to_write


def _prepare_analysis_logs(
    video_path: str,
    ranked_frames: List[RankedFrame],
    features: List[FrameFeatures],
    scores: List[FrameScore],
    vad_timestamps: Optional[List[Dict[str, float]]] = None,
) -> Tuple[Path, List[Tuple[Path, str]]]:
    video_base_name = Path(video_path).stem
    output_dir = Path("output") / video_base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_to_features = {f.frame_index: f for f in features}
    frame_to_scores = {s.frame_index: s for s in scores}

    logs_to_write: List[Tuple[Path, str]] = []

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
        output_path = output_dir / output_filename
        logs_to_write.append((output_path, json.dumps(log_data, indent=2)))

    if vad_timestamps:
        vad_log_data = {
            "speech_segments": [{"start": seg.start, "end": seg.end} for seg in vad_timestamps],
            "total_segments": len(vad_timestamps),
            "speech_end_time": vad_timestamps[-1].end if vad_timestamps else 0.0,
        }
        vad_path = output_dir / "vad_timestamps.json"
        logs_to_write.append((vad_path, json.dumps(vad_log_data, indent=2)))

    return output_dir, logs_to_write


def save_annotated_frames(
    session: "VideoSession",
    ranked_frames: List[RankedFrame],
    extractor: FeatureExtractor,
) -> None:
    output_dir, frames_to_write = _prepare_annotated_frames(session, ranked_frames, extractor)

    if not frames_to_write:
        logger.warning("No frames to save")
        return

    saved_count = 0
    failed_count = 0

    for path, data in frames_to_write:
        try:
            with open(path, "wb") as f:
                f.write(data)
            saved_count += 1
        except Exception as e:
            logger.error("Failed to write frame to %s: %s", path, e)
            failed_count += 1

    _log_save_result("frames", saved_count, failed_count, output_dir)


async def save_annotated_frames_async(
    session: "VideoSession",
    ranked_frames: List[RankedFrame],
    extractor: FeatureExtractor,
) -> None:
    output_dir, frames_to_write = _prepare_annotated_frames(session, ranked_frames, extractor)

    if not frames_to_write:
        logger.warning("No frames to save")
        return

    async def write_frame(path: Path, data: bytes) -> bool:
        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(data)
            return True
        except Exception as e:
            logger.error("Failed to write frame to %s: %s", path, e)
            return False

    results = await asyncio.gather(*[write_frame(path, data) for path, data in frames_to_write])
    saved_count = sum(results)
    failed_count = len(results) - saved_count

    _log_save_result("frames", saved_count, failed_count, output_dir)


def save_analysis_logs(
    video_path: str,
    ranked_frames: List[RankedFrame],
    features: List[FrameFeatures],
    scores: List[FrameScore],
    vad_timestamps: Optional[List[Dict[str, float]]] = None,
) -> None:
    output_dir, logs_to_write = _prepare_analysis_logs(
        video_path, ranked_frames, features, scores, vad_timestamps
    )

    if not logs_to_write:
        logger.warning("No logs to save")
        return

    saved_count = 0
    failed_count = 0

    for path, data in logs_to_write:
        try:
            with open(path, "w") as f:
                f.write(data)
            saved_count += 1
        except Exception as e:
            logger.error("Failed to write log to %s: %s", path, e)
            failed_count += 1

    _log_save_result("logs", saved_count, failed_count, output_dir)


async def save_analysis_logs_async(
    video_path: str,
    ranked_frames: List[RankedFrame],
    features: List[FrameFeatures],
    scores: List[FrameScore],
    vad_timestamps: Optional[List[Dict[str, float]]] = None,
) -> None:
    output_dir, logs_to_write = _prepare_analysis_logs(
        video_path, ranked_frames, features, scores, vad_timestamps
    )

    if not logs_to_write:
        logger.warning("No logs to save")
        return

    async def write_log(path: Path, data: str) -> bool:
        try:
            async with aiofiles.open(path, "w") as f:
                await f.write(data)
            return True
        except Exception as e:
            logger.error("Failed to write log to %s: %s", path, e)
            return False

    results = await asyncio.gather(*[write_log(path, data) for path, data in logs_to_write])
    saved_count = sum(results)
    failed_count = len(results) - saved_count

    _log_save_result("logs", saved_count, failed_count, output_dir)


def _log_save_result(item_type: str, saved_count: int, failed_count: int, output_dir: Path) -> None:
    if saved_count == 0:
        logger.warning("No %s saved", item_type)
    elif failed_count > 0:
        logger.warning(
            "Saved %d %s, %d failed to write to: %s",
            saved_count,
            item_type,
            failed_count,
            output_dir,
        )
    else:
        logger.info("Saved %d %s to: %s", saved_count, item_type, output_dir)


def _draw_landmarks(frame: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
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
