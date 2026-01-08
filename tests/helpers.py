"""Shared test constants and helper functions for SceneFlow tests.

This module contains constants and helper functions that are shared across
multiple test files. Pytest fixtures should be defined in conftest.py.
"""

import numpy as np
from typing import List

from sceneflow.shared.models import RankedFrame
from sceneflow.shared.constants import EAR, MAR, INSIGHTFACE


# =============================================================================
# MODULE-LEVEL CONSTANTS (Test Configuration)
# =============================================================================

DEFAULT_FPS = 30.0
DEFAULT_FRAME_COUNT = 300
DEFAULT_DURATION = 10.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

SMALL_WIDTH = 640
SMALL_HEIGHT = 480

DEFAULT_FRAME_INDEX = 100
DEFAULT_TIMESTAMP = 1.0
DEFAULT_SHARPNESS = 150.0
DEFAULT_FACE_CENTER = (320.0, 240.0)

TEST_VIDEO_PATH = "/test/video.mp4"
TEST_VIDEO_URL = "https://example.com/video.mp4"
TEST_OUTPUT_PATH = "/test/output.mp4"
TEST_API_KEY = "test-api-key-12345"

TEST_SPEECH_END_TIME = 5.0
TEST_BEST_TIMESTAMP = 7.5
TEST_FRAME_SCORE = 0.85
TEST_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF"

VALID_EAR = (EAR.NORMAL_MIN + EAR.NORMAL_MAX) / 2
VALID_MAR = (MAR.CLOSED_MIN + MAR.CLOSED_MAX) / 2

EAR_BELOW_VALID = EAR.MIN_VALID - 0.01
EAR_ABOVE_VALID = EAR.MAX_VALID + 0.01
EAR_SQUINTING = (EAR.MIN_VALID + EAR.NORMAL_MIN) / 2
EAR_WIDE_OPEN = (EAR.NORMAL_MAX + EAR.MAX_VALID) / 2

MAR_BELOW_VALID = MAR.MIN_VALID - 0.01
MAR_ABOVE_VALID = MAR.MAX_VALID + 0.1
MAR_SLIGHTLY_OPEN = (MAR.MIN_VALID + MAR.CLOSED_MIN) / 2
MAR_TALKING = MAR.CLOSED_MAX + 0.1

BELOW_THRESHOLD_SHARPNESS = 40.0

LEFT_EYE_INDICES = list(range(INSIGHTFACE.LEFT_EYE_START, INSIGHTFACE.LEFT_EYE_END))
RIGHT_EYE_INDICES = list(range(INSIGHTFACE.RIGHT_EYE_START, INSIGHTFACE.RIGHT_EYE_END))
MOUTH_INDICES = list(range(INSIGHTFACE.MOUTH_OUTER_START, INSIGHTFACE.MOUTH_OUTER_END))
TOTAL_LANDMARKS_REQUIRED = (
    max(max(LEFT_EYE_INDICES), max(RIGHT_EYE_INDICES), max(MOUTH_INDICES)) + 1
)


# =============================================================================
# HELPER FUNCTIONS - For creating test objects with custom parameters
# =============================================================================


def make_valid_landmarks() -> np.ndarray:
    """Create a valid landmarks array with proper eye and mouth points."""
    return _valid_landmarks_impl()


def make_invalid_eye_landmarks(eye_type: str = "left") -> np.ndarray:
    """Create landmarks with invalid (collapsed) eye points."""
    return _make_invalid_eye_landmarks_impl(eye_type=eye_type)


def make_ranked_frame(
    rank: int = 1,
    frame_index: int = 100,
    timestamp: float = 7.5,
    score: float = 0.85,
) -> RankedFrame:
    """Helper function to create a single RankedFrame object."""
    return RankedFrame(rank=rank, frame_index=frame_index, timestamp=timestamp, score=score)


def make_ranked_frames_list(
    count: int = 5,
    start_timestamp: float = 7.5,
    fps: float = 30.0,
) -> List[RankedFrame]:
    """Helper function to create a list of RankedFrame objects."""
    return [
        RankedFrame(
            rank=i + 1,
            frame_index=int((start_timestamp + i * 0.5) * fps),
            timestamp=start_timestamp + i * 0.5,
            score=1.0 - (i * 0.1),
        )
        for i in range(count)
    ]


# =============================================================================
# INTERNAL IMPLEMENTATION FUNCTIONS
# =============================================================================


def _valid_landmarks_impl() -> np.ndarray:
    """Internal helper to create valid facial landmarks."""
    landmarks = np.zeros((TOTAL_LANDMARKS_REQUIRED, 2), dtype=np.float32)

    left_eye_points = np.array(
        [
            [100, 110],
            [105, 107],
            [115, 107],
            [120, 110],
            [115, 113],
            [105, 113],
        ],
        dtype=np.float32,
    )
    for i, idx in enumerate(LEFT_EYE_INDICES):
        landmarks[idx] = left_eye_points[i]

    right_eye_points = np.array(
        [
            [180, 110],
            [185, 107],
            [195, 107],
            [200, 110],
            [195, 113],
            [185, 113],
        ],
        dtype=np.float32,
    )
    for i, idx in enumerate(RIGHT_EYE_INDICES):
        landmarks[idx] = right_eye_points[i]

    mouth_points = np.zeros((20, 2), dtype=np.float32)
    mouth_points[0] = [120, 200]
    mouth_points[2] = [140, 194]
    mouth_points[4] = [150, 193]
    mouth_points[6] = [180, 200]
    mouth_points[8] = [150, 207]
    mouth_points[10] = [140, 206]
    for i in range(20):
        if mouth_points[i].sum() == 0:
            mouth_points[i] = [150, 200]

    for i, idx in enumerate(MOUTH_INDICES):
        landmarks[idx] = mouth_points[i]

    return landmarks


def _make_invalid_eye_landmarks_impl(eye_type: str = "left") -> np.ndarray:
    """Internal helper to create landmarks with collapsed eye."""
    landmarks = np.zeros((TOTAL_LANDMARKS_REQUIRED, 2), dtype=np.float32)

    left_eye_points = np.array(
        [
            [100, 110],
            [105, 107],
            [115, 107],
            [120, 110],
            [115, 113],
            [105, 113],
        ],
        dtype=np.float32,
    )
    for i, idx in enumerate(LEFT_EYE_INDICES):
        landmarks[idx] = left_eye_points[i]

    if eye_type == "left":
        collapsed_point = [100, 110]
        for idx in LEFT_EYE_INDICES:
            landmarks[idx] = collapsed_point
    else:
        collapsed_point = [180, 110]
        for idx in RIGHT_EYE_INDICES:
            landmarks[idx] = collapsed_point

    return landmarks
