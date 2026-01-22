"""Shared pytest fixtures and mocks for SceneFlow tests."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import List

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.constants import EAR, MAR, INSIGHTFACE
from sceneflow.shared.models import (
    FrameFeatures,
    RankedFrame,
    RankingResult,
    VideoProperties,
    FaceMetrics,
)

from tests.helpers import (
    DEFAULT_FPS,
    DEFAULT_FRAME_COUNT,
    DEFAULT_DURATION,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    SMALL_WIDTH,
    SMALL_HEIGHT,
    DEFAULT_FRAME_INDEX,
    DEFAULT_TIMESTAMP,
    DEFAULT_SHARPNESS,
    DEFAULT_FACE_CENTER,
    TEST_VIDEO_PATH,
    VALID_EAR,
    VALID_MAR,
    _valid_landmarks_impl,
    make_valid_landmarks,
    make_invalid_eye_landmarks,
    make_ranked_frame as _make_ranked_frame_impl,
    make_ranked_frames_list as _make_ranked_frames_list_impl,
)


# =============================================================================
# FACTORY FIXTURES - Return functions for custom parameters
# =============================================================================


@pytest.fixture
def make_video_properties():
    def _make(
        fps: float = DEFAULT_FPS,
        frame_count: int = DEFAULT_FRAME_COUNT,
        duration: float = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> VideoProperties:
        if duration is None:
            duration = frame_count / fps
        return VideoProperties(
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            width=width,
            height=height,
        )

    return _make


@pytest.fixture
def make_video_session():
    def _make(
        video_path: str = TEST_VIDEO_PATH,
        fps: float = DEFAULT_FPS,
        frame_count: int = DEFAULT_FRAME_COUNT,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> MagicMock:
        session = MagicMock()
        session.video_path = video_path
        session.properties = VideoProperties(
            fps=fps,
            frame_count=frame_count,
            duration=frame_count / fps,
            width=width,
            height=height,
        )
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        return session

    return _make


@pytest.fixture
def make_ranked_frame():
    """Factory fixture for creating a single RankedFrame."""
    return _make_ranked_frame_impl


@pytest.fixture
def make_ranked_frames():
    """Factory fixture for creating a list of RankedFrame objects."""
    return _make_ranked_frames_list_impl


@pytest.fixture
def make_frame_features():
    def _make(
        frame_index: int = 0,
        timestamp: float = DEFAULT_TIMESTAMP,
        eye_openness: float = None,
        mouth_openness: float = None,
        face_detected: bool = True,
        sharpness: float = DEFAULT_SHARPNESS,
        face_center: tuple = DEFAULT_FACE_CENTER,
    ) -> FrameFeatures:
        if eye_openness is None:
            eye_openness = VALID_EAR
        if mouth_openness is None:
            mouth_openness = VALID_MAR
        return FrameFeatures(
            frame_index=frame_index,
            timestamp=timestamp,
            eye_openness=eye_openness,
            mouth_openness=mouth_openness,
            face_detected=face_detected,
            sharpness=sharpness,
            face_center=face_center,
        )

    return _make


@pytest.fixture
def make_face_metrics():
    def _make(
        ear: float = None,
        mar: float = None,
        sharpness: float = DEFAULT_SHARPNESS,
        center: tuple = DEFAULT_FACE_CENTER,
        detected: bool = True,
    ) -> FaceMetrics:
        if ear is None:
            ear = VALID_EAR
        if mar is None:
            mar = VALID_MAR
        return FaceMetrics(
            ear=ear,
            mar=mar,
            sharpness=sharpness,
            center=center,
            detected=detected,
        )

    return _make


@pytest.fixture
def make_mock_face():
    def _make(
        bbox: np.ndarray = None,
        det_score: float = 0.9,
        landmarks: np.ndarray = None,
        has_det_score: bool = True,
        has_landmarks: bool = True,
    ) -> Mock:
        if bbox is None:
            bbox = np.array([100.0, 100.0, 300.0, 300.0])
        face = Mock()
        face.bbox = bbox

        if has_det_score:
            face.det_score = det_score
        else:
            del face.det_score

        if has_landmarks:
            if landmarks is None:
                landmarks = _valid_landmarks_impl()
            face.landmark_2d_106 = landmarks
        else:
            face.landmark_2d_106 = None

        return face

    return _make


@pytest.fixture
def make_audio_with_energy_pattern():
    def _make(
        energy_db_per_frame: list[float],
        fps: float = DEFAULT_FPS,
        sr: int = 16000,
    ) -> np.ndarray:
        samples_per_frame = int(sr / fps)
        total_samples = len(energy_db_per_frame) * samples_per_frame
        audio = np.zeros(total_samples, dtype=np.float32)

        for frame_idx, db_level in enumerate(energy_db_per_frame):
            amplitude = 10 ** (db_level / 20)
            start_sample = frame_idx * samples_per_frame
            end_sample = start_sample + samples_per_frame
            audio[start_sample:end_sample] = amplitude

        return audio

    return _make


# =============================================================================
# DIRECT FIXTURES - Common default cases
# =============================================================================


@pytest.fixture
def video_properties():
    return VideoProperties(
        fps=DEFAULT_FPS,
        frame_count=DEFAULT_FRAME_COUNT,
        duration=DEFAULT_DURATION,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )


@pytest.fixture
def video_session():
    session = MagicMock()
    session.video_path = TEST_VIDEO_PATH
    session.properties = VideoProperties(
        fps=DEFAULT_FPS,
        frame_count=DEFAULT_FRAME_COUNT,
        duration=DEFAULT_DURATION,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    return session


@pytest.fixture
def face_metrics():
    return FaceMetrics(
        ear=VALID_EAR,
        mar=VALID_MAR,
        sharpness=DEFAULT_SHARPNESS,
        center=DEFAULT_FACE_CENTER,
        detected=True,
    )


# =============================================================================
# LANDMARK VARIANT FIXTURES
# =============================================================================


@pytest.fixture
def valid_landmarks() -> np.ndarray:
    return make_valid_landmarks()


@pytest.fixture
def invalid_eye_landmarks():
    return make_invalid_eye_landmarks(eye_type="left")


# =============================================================================
# CONFIG FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> RankingConfig:
    return RankingConfig()


@pytest.fixture
def custom_config() -> RankingConfig:
    return RankingConfig(
        eye_openness_weight=0.40,
        motion_stability_weight=0.20,
        expression_neutrality_weight=0.20,
        pose_stability_weight=0.10,
        visual_sharpness_weight=0.10,
    )


# =============================================================================
# SIMPLE LANDMARK FIXTURES (for face_metrics tests)
# =============================================================================


@pytest.fixture
def normal_eye_landmarks() -> np.ndarray:
    return np.array(
        [
            [10, 20],
            [15, 15],
            [20, 14],
            [25, 15],
            [30, 20],
            [20, 26],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def closed_eye_landmarks() -> np.ndarray:
    return np.array(
        [
            [10, 20],
            [15, 19],
            [20, 19],
            [25, 19],
            [30, 20],
            [20, 21],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def wide_eye_landmarks() -> np.ndarray:
    return np.array(
        [
            [10, 20],
            [15, 10],
            [20, 8],
            [25, 10],
            [30, 20],
            [20, 32],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def closed_mouth_landmarks() -> np.ndarray:
    landmarks = np.zeros((20, 2), dtype=np.float64)
    landmarks[0] = [10, 50]
    landmarks[2] = [20, 48]
    landmarks[4] = [30, 48]
    landmarks[6] = [50, 50]
    landmarks[8] = [30, 52]
    landmarks[10] = [20, 52]
    return landmarks


@pytest.fixture
def open_mouth_landmarks() -> np.ndarray:
    landmarks = np.zeros((20, 2), dtype=np.float64)
    landmarks[0] = [10, 50]
    landmarks[2] = [20, 45]
    landmarks[4] = [30, 45]
    landmarks[6] = [50, 50]
    landmarks[8] = [30, 58]
    landmarks[10] = [20, 58]
    return landmarks


# =============================================================================
# FRAME FEATURE FIXTURES
# =============================================================================


@pytest.fixture
def ideal_frame_features() -> FrameFeatures:
    return FrameFeatures(
        frame_index=DEFAULT_FRAME_INDEX,
        timestamp=3.33,
        eye_openness=VALID_EAR,
        mouth_openness=VALID_MAR,
        face_detected=True,
        sharpness=DEFAULT_SHARPNESS,
        face_center=DEFAULT_FACE_CENTER,
    )


@pytest.fixture
def blink_frame_features() -> FrameFeatures:
    return FrameFeatures(
        frame_index=DEFAULT_FRAME_INDEX + 1,
        timestamp=3.363,
        eye_openness=0.15,
        mouth_openness=VALID_MAR,
        face_detected=True,
        sharpness=DEFAULT_SHARPNESS,
        face_center=DEFAULT_FACE_CENTER,
    )


@pytest.fixture
def talking_frame_features() -> FrameFeatures:
    return FrameFeatures(
        frame_index=DEFAULT_FRAME_INDEX + 2,
        timestamp=3.397,
        eye_openness=VALID_EAR,
        mouth_openness=0.50,
        face_detected=True,
        sharpness=DEFAULT_SHARPNESS,
        face_center=DEFAULT_FACE_CENTER,
    )


@pytest.fixture
def blurry_frame_features() -> FrameFeatures:
    return FrameFeatures(
        frame_index=DEFAULT_FRAME_INDEX + 3,
        timestamp=3.433,
        eye_openness=VALID_EAR,
        mouth_openness=VALID_MAR,
        face_detected=True,
        sharpness=30.0,
        face_center=DEFAULT_FACE_CENTER,
    )


@pytest.fixture
def no_face_features() -> FrameFeatures:
    return FrameFeatures(
        frame_index=DEFAULT_FRAME_INDEX + 4,
        timestamp=3.466,
        eye_openness=EAR.DEFAULT,
        mouth_openness=MAR.DEFAULT,
        face_detected=False,
        sharpness=0.0,
        face_center=(0.0, 0.0),
    )


@pytest.fixture
def sample_features_list(
    ideal_frame_features,
    blink_frame_features,
    talking_frame_features,
    blurry_frame_features,
) -> List[FrameFeatures]:
    return [
        ideal_frame_features,
        blink_frame_features,
        talking_frame_features,
        blurry_frame_features,
    ]


@pytest.fixture
def consistent_features_sequence() -> List[FrameFeatures]:
    return [
        FrameFeatures(
            frame_index=i,
            timestamp=i * 0.033,
            eye_openness=VALID_EAR,
            mouth_openness=VALID_MAR,
            face_detected=True,
            sharpness=DEFAULT_SHARPNESS,
            face_center=(320.0 + i * 2, 240.0),
        )
        for i in range(5)
    ]


@pytest.fixture
def inconsistent_features_sequence() -> List[FrameFeatures]:
    centers = [(100, 100), (400, 400), (150, 350), (380, 120), (250, 250)]
    return [
        FrameFeatures(
            frame_index=i,
            timestamp=i * 0.033,
            eye_openness=VALID_EAR,
            mouth_openness=VALID_MAR,
            face_detected=True,
            sharpness=DEFAULT_SHARPNESS,
            face_center=centers[i],
        )
        for i in range(5)
    ]


# =============================================================================
# VIDEO MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_frame() -> np.ndarray:
    return np.random.randint(0, 255, (SMALL_HEIGHT, SMALL_WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def mock_video_capture() -> MagicMock:
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        0: DEFAULT_FRAME_COUNT,
        3: DEFAULT_WIDTH,
        4: DEFAULT_HEIGHT,
        5: DEFAULT_FPS,
        7: DEFAULT_FRAME_COUNT,
    }.get(prop, 0)

    frame = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    cap.set.return_value = True
    cap.release.return_value = None

    return cap


# =============================================================================
# INSIGHTFACE MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_face_106_landmarks() -> Mock:
    face = Mock()
    face.det_score = INSIGHTFACE.MIN_FACE_CONFIDENCE + 0.49

    landmarks = np.zeros((106, 2), dtype=np.float64)
    landmarks[INSIGHTFACE.LEFT_EYE_START : INSIGHTFACE.LEFT_EYE_END] = [
        [10, 20],
        [15, 15],
        [20, 14],
        [25, 15],
        [30, 20],
        [20, 26],
    ]
    landmarks[INSIGHTFACE.RIGHT_EYE_START : INSIGHTFACE.RIGHT_EYE_END] = [
        [50, 20],
        [55, 15],
        [60, 14],
        [65, 15],
        [70, 20],
        [60, 26],
    ]
    for i, idx in enumerate(range(INSIGHTFACE.MOUTH_OUTER_START, INSIGHTFACE.MOUTH_OUTER_END)):
        landmarks[idx] = [40 + i * 2, 80 + (i % 4)]

    face.landmark_2d_106 = landmarks
    face.bbox = np.array([100, 100, 500, 500])

    return face


@pytest.fixture
def mock_face_no_landmarks() -> Mock:
    face = Mock()
    face.det_score = INSIGHTFACE.MIN_FACE_CONFIDENCE + 0.49
    face.landmark_2d_106 = None
    face.bbox = np.array([100, 100, 500, 500])
    return face


@pytest.fixture
def mock_insightface_app(mock_face_106_landmarks) -> Mock:
    app = Mock()
    app.get.return_value = [mock_face_106_landmarks]
    app.prepare.return_value = None
    return app


# =============================================================================
# VAD MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_vad_model() -> Mock:
    return Mock()


@pytest.fixture
def mock_speech_timestamps() -> List[dict]:
    return [
        {"start": 0, "end": 160000},
        {"start": 176000, "end": 256000},
    ]


# =============================================================================
# OPENAI MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_openai_response() -> Mock:
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "1"
    return response


@pytest.fixture
def mock_openai_client(mock_openai_response) -> Mock:
    client = Mock()
    client.chat.completions.create.return_value = mock_openai_response
    return client


# =============================================================================
# RANKING RESULT FIXTURES
# =============================================================================


@pytest.fixture
def sample_ranked_frames() -> List[RankedFrame]:
    return [
        RankedFrame(rank=1, frame_index=100, timestamp=3.33, score=0.95),
        RankedFrame(rank=2, frame_index=150, timestamp=5.00, score=0.90),
        RankedFrame(rank=3, frame_index=200, timestamp=6.67, score=0.85),
        RankedFrame(rank=4, frame_index=250, timestamp=8.33, score=0.80),
        RankedFrame(rank=5, frame_index=280, timestamp=9.33, score=0.75),
    ]


@pytest.fixture
def sample_ranking_result(sample_ranked_frames, sample_features_list) -> RankingResult:
    return RankingResult(
        ranked_frames=sample_ranked_frames,
        features=sample_features_list,
        scores=None,
    )
