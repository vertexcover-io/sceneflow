"""
Energy-based speech end refinement.

Refines VAD timestamps by detecting sudden drops in audio energy levels.
"""
import librosa
import numpy as np
import logging

from sceneflow.shared.models import EnergyRefinementResult
from sceneflow.utils.video import get_video_properties

logger = logging.getLogger(__name__)


class EnergyRefiner:
    """Refines VAD speech end timestamps using audio energy analysis."""

    def __init__(
        self,
        threshold_db: float = 10.0,
        lookback_frames: int = 20,
        min_silence_frames: int = 1,
    ):
        """
        Initialize the energy refiner.

        Args:
            threshold_db: Minimum dB drop to consider speech end (default: 8.0)
            lookback_frames: Maximum frames to search backward (default: 20)
            min_silence_frames: Consecutive low-energy frames required (default: 2)
        """
        self.threshold_db = threshold_db
        self.lookback_frames = lookback_frames
        self.min_silence_frames = min_silence_frames

    def refine_speech_end(
        self, vad_timestamp: float, video_path: str
    ) -> EnergyRefinementResult:
        """
        Refine VAD timestamp by finding actual speech end via energy drop.

        Args:
            vad_timestamp: Rough timestamp from Silero VAD
            video_path: Path to video file

        Returns:
            EnergyRefinementResult dataclass with refined timestamp and metadata
        """
        props = get_video_properties(video_path)
        fps = props.fps

        # Convert VAD timestamp to frame
        vad_frame = int(vad_timestamp * fps)

        # Load audio
        y, sr = librosa.load(video_path, sr=None)

        # Define search range
        start_frame = max(0, vad_frame - self.lookback_frames)
        end_frame = vad_frame

        # Extract energy levels for all frames in range
        energy_levels = self._extract_energy_levels(y, sr, fps, start_frame, end_frame)

        # Find speech end frame
        refined_frame = self._find_speech_end_frame(
            energy_levels, start_frame, end_frame
        )

        # Calculate metadata
        # If no refinement occurred, use original VAD timestamp to preserve precision
        # Otherwise use frame-based timestamp
        if refined_frame == vad_frame:
            refined_timestamp = vad_timestamp  # Preserve original precision
            energy_drop = 0.0
        else:
            refined_timestamp = refined_frame / fps
            energy_drop = energy_levels.get(refined_frame - 1, 0) - energy_levels.get(
                refined_frame, 0
            )

        result = EnergyRefinementResult(
            refined_timestamp=refined_timestamp,
            vad_frame=vad_frame,
            vad_timestamp=vad_timestamp,
            refined_frame=refined_frame,
            energy_drop_db=energy_drop,
            frames_adjusted=vad_frame - refined_frame,
            energy_levels=energy_levels
        )

        logger.info(f"VAD timestamp: {vad_timestamp:.4f}s (frame {vad_frame})")
        if refined_frame != vad_frame:
            logger.info(
                f"Refined timestamp: {refined_timestamp:.4f}s (frame {refined_frame})"
            )
            logger.info(f"Adjusted by {vad_frame - refined_frame} frames backward")
            logger.info(f"Energy drop detected: {energy_drop:.2f} dB")
        else:
            logger.info("No refinement needed - using original VAD timestamp")

        return result

    def _extract_energy_levels(
        self, y: np.ndarray, sr: int, fps: float, start_frame: int, end_frame: int
    ) -> dict[int, float]:
        """
        Extract dB energy levels for frame range.

        Args:
            y: Audio samples
            sr: Sample rate
            fps: Video frame rate
            start_frame: Start frame number
            end_frame: End frame number

        Returns:
            Dict mapping frame_number -> dB_level
        """
        energy_levels = {}
        samples_per_frame = int(sr / fps)

        for frame_num in range(start_frame, end_frame + 1):
            # Convert frame to audio sample index
            sample_idx = int((frame_num / fps) * sr)
            start_sample = max(0, sample_idx)
            end_sample = min(len(y), sample_idx + samples_per_frame)

            # Extract segment and calculate RMS energy
            segment = y[start_sample:end_sample]
            rms = np.sqrt(np.mean(segment**2))
            db = 20 * np.log10(rms + 1e-10)

            energy_levels[frame_num] = db

        return energy_levels

    def _find_speech_end_frame(
        self, energy_levels: dict[int, float], start_frame: int, end_frame: int
    ) -> int:
        """
        Find frame where speech ends based on energy drop.

        Args:
            energy_levels: Dict of frame -> dB level
            start_frame: Search start frame
            end_frame: Search end frame (VAD frame)

        Returns:
            Frame number where speech ends
        """
        # Search backward from VAD frame
        for frame in range(end_frame, start_frame, -1):
            if frame - 1 not in energy_levels:
                continue

            current_db = energy_levels[frame]
            prev_db = energy_levels[frame - 1]
            drop = prev_db - current_db

            # Check if we found a significant drop
            if drop >= self.threshold_db:
                # Verify silence persists for min_silence_frames
                if self._verify_silence_persists(energy_levels, frame, end_frame):
                    return frame

        # No significant drop found, return VAD frame
        logger.warning(
            f"No energy drop >= {self.threshold_db} dB found, using VAD timestamp"
        )
        return end_frame

    def _verify_silence_persists(
        self, energy_levels: dict[int, float], start_frame: int, end_frame: int
    ) -> bool:
        """
        Verify that silence persists for minimum required frames.

        Args:
            energy_levels: Dict of frame -> dB level
            start_frame: Frame where potential speech end detected
            end_frame: Last frame to check

        Returns:
            True if silence persists for min_silence_frames
        """
        silence_threshold = energy_levels[start_frame]
        consecutive_silence = 0

        for frame in range(start_frame, min(end_frame + 1, start_frame + self.min_silence_frames + 1)):
            if frame not in energy_levels:
                break

            if energy_levels[frame] <= silence_threshold + 3:  # Allow 3 dB tolerance
                consecutive_silence += 1
            else:
                return False

        return consecutive_silence >= self.min_silence_frames
