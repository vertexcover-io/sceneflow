"""Tests for SpeechDetector with mocked Silero VAD."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from sceneflow.detection.speech_detector import SpeechDetector
from sceneflow.shared.exceptions import AudioLoadError
from sceneflow.shared.constants import VAD
from sceneflow.shared.models import EnergyRefinementResult

from tests.helpers import DEFAULT_FPS, DEFAULT_DURATION, DEFAULT_WIDTH, DEFAULT_HEIGHT
from sceneflow.shared.models import VideoProperties


TEST_VIDEO_PATH = "/path/to/test_video.mp4"
SINGLE_SEGMENT_START = 0.5
SINGLE_SEGMENT_END = 5.0
MULTI_SEGMENT_ENDS = [2.0, 4.0, 8.5]
SHORT_SEGMENT_DURATION = VAD.MIN_SEGMENT_DURATION_FOR_FULL_CONFIDENCE / 2
LONG_SEGMENT_DURATION = VAD.MIN_SEGMENT_DURATION_FOR_FULL_CONFIDENCE * 2
REFINED_TIMESTAMP = 4.8
ENERGY_THRESHOLD_DB = 8.0
ENERGY_LOOKBACK_FRAMES = 20


@pytest.fixture
def mock_video_session():
    session = Mock()
    session.video_path = TEST_VIDEO_PATH
    session.properties = VideoProperties(
        fps=DEFAULT_FPS,
        frame_count=int(DEFAULT_DURATION * DEFAULT_FPS),
        duration=DEFAULT_DURATION,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )
    audio_samples = int(DEFAULT_DURATION * VAD.TARGET_SAMPLE_RATE)
    session.get_audio.return_value = (
        np.zeros(audio_samples, dtype=np.float32),
        VAD.TARGET_SAMPLE_RATE,
    )
    return session


@pytest.fixture
def speech_detector(mock_vad_model):
    with patch("sceneflow.detection.speech_detector.load_silero_vad", return_value=mock_vad_model):
        detector = SpeechDetector()
    return detector


class TestGetSpeechEndTime:
    def test_normal_video_with_speech_returns_valid_result(
        self, speech_detector, mock_video_session
    ):
        speech_timestamps = [{"start": SINGLE_SEGMENT_START, "end": SINGLE_SEGMENT_END}]

        with patch(
            "sceneflow.detection.speech_detector.get_speech_timestamps",
            return_value=speech_timestamps,
        ):
            with patch("sceneflow.detection.energy_refiner.refine_speech_end") as mock_refine:
                mock_refine.return_value = EnergyRefinementResult(
                    refined_timestamp=SINGLE_SEGMENT_END,
                    vad_frame=int(SINGLE_SEGMENT_END * DEFAULT_FPS),
                    vad_timestamp=SINGLE_SEGMENT_END,
                    refined_frame=int(SINGLE_SEGMENT_END * DEFAULT_FPS),
                    energy_drop_db=0.0,
                    energy_levels={},
                )
                end_time, confidence = speech_detector.get_speech_end_time(mock_video_session)

        assert end_time == SINGLE_SEGMENT_END
        assert 0.0 < confidence <= 1.0

    def test_no_speech_detected_returns_zero(self, speech_detector, mock_video_session):
        with patch("sceneflow.detection.speech_detector.get_speech_timestamps", return_value=[]):
            with patch("sceneflow.detection.energy_refiner.refine_speech_end") as mock_refine:
                mock_refine.return_value = EnergyRefinementResult(
                    refined_timestamp=0.0,
                    vad_frame=0,
                    vad_timestamp=0.0,
                    refined_frame=0,
                    energy_drop_db=0.0,
                    energy_levels={},
                )
                end_time, confidence = speech_detector.get_speech_end_time(mock_video_session)

        assert end_time == 0.0
        assert confidence == 0.0

    def test_audio_load_failure_raises_audio_load_error(self, speech_detector, mock_video_session):
        mock_video_session.get_audio.side_effect = Exception("Failed to decode audio")

        with pytest.raises(AudioLoadError):
            speech_detector.get_speech_end_time(mock_video_session)

    def test_energy_refinement_disabled_skips_refiner(self, speech_detector, mock_video_session):
        speech_timestamps = [{"start": SINGLE_SEGMENT_START, "end": SINGLE_SEGMENT_END}]

        with patch(
            "sceneflow.detection.speech_detector.get_speech_timestamps",
            return_value=speech_timestamps,
        ):
            with patch("sceneflow.detection.energy_refiner.refine_speech_end") as mock_refine:
                end_time, _ = speech_detector.get_speech_end_time(
                    mock_video_session,
                    use_energy_refinement=False,
                )

        mock_refine.assert_not_called()
        assert end_time == SINGLE_SEGMENT_END

    def test_energy_refinement_enabled_calls_refiner_and_returns_refined_result(
        self, speech_detector, mock_video_session
    ):
        speech_timestamps = [{"start": SINGLE_SEGMENT_START, "end": SINGLE_SEGMENT_END}]
        vad_frame = int(SINGLE_SEGMENT_END * DEFAULT_FPS)
        refined_frame = int(REFINED_TIMESTAMP * DEFAULT_FPS)

        with patch(
            "sceneflow.detection.speech_detector.get_speech_timestamps",
            return_value=speech_timestamps,
        ):
            with patch("sceneflow.detection.energy_refiner.refine_speech_end") as mock_refine:
                mock_refine.return_value = EnergyRefinementResult(
                    refined_timestamp=REFINED_TIMESTAMP,
                    vad_frame=vad_frame,
                    vad_timestamp=SINGLE_SEGMENT_END,
                    refined_frame=refined_frame,
                    energy_drop_db=10.0,
                    energy_levels={},
                )
                end_time, _ = speech_detector.get_speech_end_time(
                    mock_video_session,
                    use_energy_refinement=True,
                    energy_threshold_db=ENERGY_THRESHOLD_DB,
                    energy_lookback_frames=ENERGY_LOOKBACK_FRAMES,
                )

        mock_refine.assert_called_once_with(
            session=mock_video_session,
            vad_timestamp=SINGLE_SEGMENT_END,
            threshold_db=ENERGY_THRESHOLD_DB,
            lookback_frames=ENERGY_LOOKBACK_FRAMES,
        )
        assert end_time == REFINED_TIMESTAMP

    def test_multiple_speech_segments_returns_end_of_last_segment(
        self, speech_detector, mock_video_session
    ):
        speech_timestamps = [
            {"start": 0.5, "end": MULTI_SEGMENT_ENDS[0]},
            {"start": 2.5, "end": MULTI_SEGMENT_ENDS[1]},
            {"start": 6.0, "end": MULTI_SEGMENT_ENDS[2]},
        ]
        expected_end = MULTI_SEGMENT_ENDS[-1]

        with patch(
            "sceneflow.detection.speech_detector.get_speech_timestamps",
            return_value=speech_timestamps,
        ):
            with patch("sceneflow.detection.energy_refiner.refine_speech_end") as mock_refine:
                mock_refine.return_value = EnergyRefinementResult(
                    refined_timestamp=expected_end,
                    vad_frame=int(expected_end * DEFAULT_FPS),
                    vad_timestamp=expected_end,
                    refined_frame=int(expected_end * DEFAULT_FPS),
                    energy_drop_db=0.0,
                    energy_levels={},
                )
                end_time, _ = speech_detector.get_speech_end_time(mock_video_session)

        assert end_time == expected_end

    def test_short_segment_returns_confidence_less_than_one(
        self, speech_detector, mock_video_session
    ):
        segment_start = 1.0
        segment_end = segment_start + SHORT_SEGMENT_DURATION
        speech_timestamps = [{"start": segment_start, "end": segment_end}]

        with patch(
            "sceneflow.detection.speech_detector.get_speech_timestamps",
            return_value=speech_timestamps,
        ):
            with patch("sceneflow.detection.energy_refiner.refine_speech_end") as mock_refine:
                mock_refine.return_value = EnergyRefinementResult(
                    refined_timestamp=segment_end,
                    vad_frame=int(segment_end * DEFAULT_FPS),
                    vad_timestamp=segment_end,
                    refined_frame=int(segment_end * DEFAULT_FPS),
                    energy_drop_db=0.0,
                    energy_levels={},
                )
                _, confidence = speech_detector.get_speech_end_time(mock_video_session)

        expected_confidence = SHORT_SEGMENT_DURATION / VAD.MIN_SEGMENT_DURATION_FOR_FULL_CONFIDENCE
        assert confidence == pytest.approx(expected_confidence)
        assert confidence < 1.0

    def test_long_segment_returns_confidence_capped_at_one(
        self, speech_detector, mock_video_session
    ):
        segment_start = 1.0
        segment_end = segment_start + LONG_SEGMENT_DURATION
        speech_timestamps = [{"start": segment_start, "end": segment_end}]

        with patch(
            "sceneflow.detection.speech_detector.get_speech_timestamps",
            return_value=speech_timestamps,
        ):
            with patch("sceneflow.detection.energy_refiner.refine_speech_end") as mock_refine:
                mock_refine.return_value = EnergyRefinementResult(
                    refined_timestamp=segment_end,
                    vad_frame=int(segment_end * DEFAULT_FPS),
                    vad_timestamp=segment_end,
                    refined_frame=int(segment_end * DEFAULT_FPS),
                    energy_drop_db=0.0,
                    energy_levels={},
                )
                _, confidence = speech_detector.get_speech_end_time(mock_video_session)

        assert confidence == 1.0
