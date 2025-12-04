"""Utilities for analyzing speech segments to find optimal cut points.

This module provides functions to analyze VAD (Voice Activity Detection) segments
and determine optimal cut points based on silence patterns and segment completeness.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def find_clean_ending_by_silence(
    vad_segments: List[Dict[str, float]],
    video_duration: float,
    incomplete_threshold: float = 0.5,
    min_silence_duration: float = 0.3
) -> float:
    """
    Find optimal cut point based on silence patterns and segment completeness.

    Searches backwards through VAD segments to find the best timestamp that:
    1. Has sufficient silence after it (min_silence_duration)
    2. Is not too close to the video end (incomplete_threshold)

    Args:
        vad_segments: Speech segments from VAD detection, each with 'start' and 'end'
        video_duration: Total video duration in seconds
        incomplete_threshold: Maximum gap (in seconds) between segment end
                            and video end to consider segment incomplete. Default: 0.5s
        min_silence_duration: Minimum silence duration required after a segment
                            to consider it a clean cut point. Default: 0.3s

    Returns:
        Optimal cut point in seconds

    Example:
        >>> # Video with multiple segments and silence gaps
        >>> vad_segments = [
        ...     {"start": 0.5, "end": 5.0},
        ...     {"start": 5.8, "end": 10.0},
        ...     {"start": 10.5, "end": 14.8}  # Too close to 15.0s end
        ... ]
        >>> cut_point = find_clean_ending_by_silence(vad_segments, 15.0)
        >>> cut_point
        10.0  # Uses segment with good silence after it

    Logic:
        1. Start from the last segment and work backwards
        2. For each segment, check if it's too close to video end
        3. If too close, find next segment with sufficient silence after it
        4. Return the first valid timestamp found
    """
    if not vad_segments:
        logger.warning("No VAD segments provided, returning 0.0")
        return 0.0

    if len(vad_segments) == 1:
        logger.debug("Only one speech segment, using its end time")
        return vad_segments[0]["end"]

    # Search backwards through segments to find best cut point
    for i in range(len(vad_segments) - 1, -1, -1):
        current_seg = vad_segments[i]

        # Calculate gap to video end (or next segment if not the last)
        if i == len(vad_segments) - 1:
            gap_after = video_duration - current_seg["end"]
        else:
            gap_after = vad_segments[i + 1]["start"] - current_seg["end"]

        logger.debug(
            "Checking segment %d: %.4f-%.4fs, gap after: %.4fs",
            i,
            current_seg["start"],
            current_seg["end"],
            gap_after
        )

        # For the last segment, check if it's too close to video end
        if i == len(vad_segments) - 1:
            if gap_after < incomplete_threshold:
                logger.debug(
                    "Last segment too close to end (gap: %.2fs < threshold: %.2fs), "
                    "searching for better cut point",
                    gap_after,
                    incomplete_threshold
                )
                continue

        # Check if this segment has sufficient silence after it
        if gap_after >= min_silence_duration:
            logger.info(
                "Found optimal cut point at %.2fs (segment %d) with %.2fs silence after",
                current_seg["end"],
                i,
                gap_after
            )
            return current_seg["end"]
        else:
            logger.debug(
                "Segment %d has insufficient silence (%.2fs < %.2fs), continuing search",
                i,
                gap_after,
                min_silence_duration
            )

    # Fallback: if no segment meets criteria, use the last segment's end
    fallback_time = vad_segments[-1]["end"]
    logger.warning(
        "No optimal cut point found with criteria, falling back to last segment at %.2fs",
        fallback_time
    )
    return fallback_time
