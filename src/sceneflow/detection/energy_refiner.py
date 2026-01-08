"""Energy-based speech end refinement."""

import numpy as np
import logging

from sceneflow.shared.models import EnergyRefinementResult

from sceneflow.utils.video import VideoSession

logger = logging.getLogger(__name__)


def _build_refinement_result(
    refined_frame: int,
    vad_frame: int,
    vad_timestamp: float,
    fps: float,
    energy_levels: dict[int, float],
) -> EnergyRefinementResult:
    if refined_frame == vad_frame:
        refined_timestamp = vad_timestamp
        energy_drop = 0.0
    else:
        refined_timestamp = refined_frame / fps
        energy_drop = energy_levels.get(refined_frame - 1, 0) - energy_levels.get(refined_frame, 0)

    result = EnergyRefinementResult(
        refined_timestamp=refined_timestamp,
        vad_frame=vad_frame,
        vad_timestamp=vad_timestamp,
        refined_frame=refined_frame,
        energy_drop_db=energy_drop,
        energy_levels=energy_levels,
    )

    if refined_frame != vad_frame:
        logger.debug(
            "Energy refinement: VAD %.4fs → Refined %.4fs (%d frames, %.2f dB drop)",
            vad_timestamp,
            refined_timestamp,
            vad_frame - refined_frame,
            energy_drop,
        )
    else:
        logger.debug("Energy refinement: no adjustment needed")

    return result


def refine_speech_end(
    session: "VideoSession",
    vad_timestamp: float,
    threshold_db: float = 8.0,
    lookback_frames: int = 10,
    min_silence_frames: int = 3,
) -> EnergyRefinementResult:
    """Refine VAD timestamp by detecting sudden drops in audio energy levels."""
    fps = session.properties.fps
    vad_frame = int(vad_timestamp * fps)

    y, sr = session.get_audio(sr=None)

    start_frame = max(0, vad_frame - lookback_frames)
    end_frame = vad_frame

    energy_levels = _extract_energy_levels(y, sr, fps, start_frame, end_frame)

    refined_frame = _find_speech_end_frame(
        energy_levels, start_frame, end_frame, threshold_db, min_silence_frames
    )

    return _build_refinement_result(refined_frame, vad_frame, vad_timestamp, fps, energy_levels)


def _extract_energy_levels(
    y: np.ndarray, sr: int, fps: float, start_frame: int, end_frame: int
) -> dict[int, float]:
    """Extract dB energy levels for frame range."""
    energy_levels = {}
    samples_per_frame = int(sr / fps)

    for frame_num in range(start_frame, end_frame + 1):
        sample_idx = int((frame_num / fps) * sr)
        start_sample = max(0, sample_idx)
        end_sample = min(len(y), sample_idx + samples_per_frame)

        segment = y[start_sample:end_sample]
        rms = np.sqrt(np.mean(segment**2))
        db = 20 * np.log10(rms + 1e-10)

        energy_levels[frame_num] = db

    return energy_levels


def _find_speech_end_frame(
    energy_levels: dict[int, float],
    start_frame: int,
    end_frame: int,
    threshold_db: float,
    min_silence_frames: int,
) -> int:
    """Find frame where speech ends based on energy drop near VAD timestamp."""
    candidate_frames = []

    for frame in range(end_frame, start_frame, -1):
        if frame - 1 not in energy_levels:
            continue

        current_db = energy_levels[frame]
        prev_db = energy_levels[frame - 1]
        drop = prev_db - current_db

        if drop >= threshold_db:
            if _verify_continuous_silence(energy_levels, frame, end_frame, min_silence_frames):
                candidate_frames.append(frame)

    if candidate_frames:
        refined_frame = max(candidate_frames)
        return refined_frame

    logger.warning(f"No energy drop >= {threshold_db} dB found, using VAD timestamp")
    return end_frame


def _verify_continuous_silence(
    energy_levels: dict[int, float],
    drop_frame: int,
    vad_frame: int,
    min_silence_frames: int,
    max_deviation_db: float = 2.0,
) -> bool:
    """Verify energy STAYS LOW from drop_frame to vad_frame."""
    silence_threshold = energy_levels[drop_frame]

    for frame in range(drop_frame, vad_frame + 1):
        if frame not in energy_levels:
            continue

        if energy_levels[frame] > silence_threshold + max_deviation_db:
            return False

    consecutive_silence = 0
    check_end = min(vad_frame + 1, drop_frame + min_silence_frames + 2)

    for frame in range(drop_frame, check_end):
        if frame not in energy_levels:
            break

        if energy_levels[frame] <= silence_threshold + max_deviation_db:
            consecutive_silence += 1
        else:
            break

    return consecutive_silence >= min_silence_frames
