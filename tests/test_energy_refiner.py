"""Tests for energy-based speech end refinement."""

import pytest
from unittest.mock import Mock

from sceneflow.shared.models import EnergyRefinementResult
from sceneflow.detection.energy_refiner import refine_speech_end

from tests.helpers import DEFAULT_FPS, DEFAULT_WIDTH, DEFAULT_HEIGHT


def create_mock_session(
    energy_db_per_frame: list[float],
    make_audio_with_energy_pattern,
    make_video_properties,
    fps: float = DEFAULT_FPS,
    sr: int = 16000,
) -> Mock:
    audio = make_audio_with_energy_pattern(energy_db_per_frame, fps, sr)

    session = Mock()
    session.properties = make_video_properties(
        fps=fps,
        frame_count=len(energy_db_per_frame),
        duration=len(energy_db_per_frame) / fps,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )
    session.get_audio.return_value = (audio, sr)

    return session


class TestRefineSpeechEnd:
    def test_finds_energy_drop_and_refines_timestamp(
        self, make_audio_with_energy_pattern, make_video_properties
    ):
        energy_pattern = [-20.0] * 10 + [-35.0] * 5
        session = create_mock_session(
            energy_pattern, make_audio_with_energy_pattern, make_video_properties, fps=DEFAULT_FPS
        )

        vad_timestamp = 14 / DEFAULT_FPS

        result = refine_speech_end(
            session=session,
            vad_timestamp=vad_timestamp,
            threshold_db=8.0,
            lookback_frames=10,
            min_silence_frames=3,
        )

        assert isinstance(result, EnergyRefinementResult)
        assert result.refined_frame == 10
        assert result.refined_timestamp == pytest.approx(10 / DEFAULT_FPS, abs=0.001)
        assert result.vad_frame == 14
        assert result.energy_drop_db > 0

    def test_falls_back_to_vad_when_no_drop_found(
        self, make_audio_with_energy_pattern, make_video_properties
    ):
        energy_pattern = [-20.0 - i for i in range(15)]
        session = create_mock_session(
            energy_pattern, make_audio_with_energy_pattern, make_video_properties, fps=DEFAULT_FPS
        )
        vad_frame = 14

        vad_timestamp = vad_frame / DEFAULT_FPS

        result = refine_speech_end(
            session=session,
            vad_timestamp=vad_timestamp,
            threshold_db=8.0,
            lookback_frames=10,
            min_silence_frames=3,
        )

        assert result.refined_frame == vad_frame
        assert result.refined_timestamp == pytest.approx(vad_timestamp, abs=0.001)
        assert result.energy_drop_db == 0.0

    def test_rejects_drop_when_silence_does_not_persist(
        self, make_audio_with_energy_pattern, make_video_properties
    ):
        energy_pattern = [-20.0] * 8 + [-35.0] + [-20.0] * 6
        session = create_mock_session(
            energy_pattern, make_audio_with_energy_pattern, make_video_properties, fps=DEFAULT_FPS
        )

        vad_timestamp = 14 / DEFAULT_FPS

        result = refine_speech_end(
            session=session,
            vad_timestamp=vad_timestamp,
            threshold_db=8.0,
            lookback_frames=10,
            min_silence_frames=3,
        )

        assert result.refined_frame == 14
        assert result.energy_drop_db == 0.0

    def test_threshold_boundary_just_below_threshold(
        self, make_audio_with_energy_pattern, make_video_properties
    ):
        energy_pattern = [-20.0] * 10 + [-27.9] * 5
        session = create_mock_session(
            energy_pattern, make_audio_with_energy_pattern, make_video_properties, fps=DEFAULT_FPS
        )

        vad_timestamp = 14 / DEFAULT_FPS

        result = refine_speech_end(
            session=session,
            vad_timestamp=vad_timestamp,
            threshold_db=8.0,
            lookback_frames=10,
            min_silence_frames=3,
        )

        assert result.refined_frame == 14
        assert result.energy_drop_db == 0.0

    def test_multiple_valid_drops_returns_latest(
        self, make_audio_with_energy_pattern, make_video_properties
    ):
        energy_pattern = [-20.0] * 5 + [-35.0] * 3 + [-20.0] * 2 + [-35.0] * 5
        session = create_mock_session(
            energy_pattern, make_audio_with_energy_pattern, make_video_properties, fps=DEFAULT_FPS
        )

        vad_timestamp = 14 / DEFAULT_FPS

        result = refine_speech_end(
            session=session,
            vad_timestamp=vad_timestamp,
            threshold_db=8.0,
            lookback_frames=12,
            min_silence_frames=3,
        )

        assert result.refined_frame == 10
        assert result.refined_timestamp == pytest.approx(10 / DEFAULT_FPS, abs=0.001)

    def test_vad_at_frame_zero_returns_vad_timestamp(
        self, make_audio_with_energy_pattern, make_video_properties
    ):
        energy_pattern = [-35.0] * 5
        session = create_mock_session(
            energy_pattern, make_audio_with_energy_pattern, make_video_properties, fps=DEFAULT_FPS
        )

        vad_timestamp = 0.0

        result = refine_speech_end(
            session=session,
            vad_timestamp=vad_timestamp,
            threshold_db=8.0,
            lookback_frames=10,
            min_silence_frames=3,
        )

        assert result.refined_frame == 0
        assert result.refined_timestamp == 0.0
        assert result.vad_frame == 0
