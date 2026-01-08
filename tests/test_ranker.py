"""Tests for CutPointRanker - rank_frames and get_detailed_scores methods."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np

from sceneflow.shared.models import FrameScore, RankingResult
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.shared.models import FaceMetrics
from sceneflow.shared.constants import EAR

from tests.helpers import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    VALID_EAR,
    DEFAULT_SHARPNESS,
    DEFAULT_FACE_CENTER,
)


FIRST_RANK = 1


@pytest.fixture
def mock_session(make_video_session):
    return make_video_session()


@pytest.fixture
def ranker():
    with patch("sceneflow.core.ranker.FeatureExtractor") as mock_extractor_class:
        mock_extractor_class.return_value = MagicMock()
        from sceneflow.core.ranker import CutPointRanker

        ranker_instance = CutPointRanker()
        yield ranker_instance


@pytest.fixture
def make_test_face_metrics():
    def _make(
        ear: float = None,
        mar: float = None,
        sharpness: float = DEFAULT_SHARPNESS,
        center: tuple = DEFAULT_FACE_CENTER,
        detected: bool = True,
    ):
        if ear is None:
            ear = VALID_EAR
        if mar is None:
            mar = (EAR.NORMAL_MIN + EAR.NORMAL_MAX) / 2
        return FaceMetrics(
            ear=ear,
            mar=mar,
            sharpness=sharpness,
            center=center,
            detected=detected,
        )

    return _make


def generate_dummy_frames(start_frame, end_frame, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH):
    frames = []
    for idx in range(start_frame, end_frame + 1):
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        frames.append((idx, dummy_frame))
    return frames


class TestRankFramesBasicFunctionality:
    def test_returns_ranked_frames_sorted_by_score_descending(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 5.0
        end_time = 6.0
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)
        mock_session.iterate_frames.return_value = iter(frames)

        varying_ears = [0.20, VALID_EAR, 0.15, VALID_EAR + 0.01, 0.10]

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            side_effect=[
                make_test_face_metrics(ear=varying_ears[i % len(varying_ears)])
                for i in range(len(frames))
            ],
        ):
            result = ranker.rank_frames(mock_session, start_time, end_time)

        assert isinstance(result, RankingResult)
        assert len(result.ranked_frames) == len(frames)

        scores = [rf.score for rf in result.ranked_frames]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_start_at_one_and_increment_consecutively(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 5.0
        end_time = 5.5
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)
        mock_session.iterate_frames.return_value = iter(frames)

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            return_value=make_test_face_metrics(),
        ):
            result = ranker.rank_frames(mock_session, start_time, end_time)

        ranks = [rf.rank for rf in result.ranked_frames]
        expected_ranks = list(range(FIRST_RANK, len(frames) + FIRST_RANK))
        assert ranks == expected_ranks

    def test_timestamp_equals_frame_index_divided_by_fps(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 2.0
        end_time = 2.5
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)
        mock_session.iterate_frames.return_value = iter(frames)

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            return_value=make_test_face_metrics(),
        ):
            result = ranker.rank_frames(mock_session, start_time, end_time)

        for ranked_frame in result.ranked_frames:
            expected_timestamp = ranked_frame.frame_index / DEFAULT_FPS
            assert ranked_frame.timestamp == pytest.approx(expected_timestamp)

    def test_ranking_result_contains_features_and_scores(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 3.0
        end_time = 3.2
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)
        mock_session.iterate_frames.return_value = iter(frames)

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            return_value=make_test_face_metrics(),
        ):
            result = ranker.rank_frames(mock_session, start_time, end_time)

        assert result.features is not None
        assert result.scores is not None
        assert len(result.features) == len(frames)
        assert len(result.scores) == len(frames)
        assert len(result.ranked_frames) == len(frames)


class TestRankFramesEdgeCases:
    def test_single_frame_returns_one_ranked_frame_with_rank_one(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 5.0
        end_time = 5.0
        frame_idx = int(start_time * DEFAULT_FPS)

        frames = generate_dummy_frames(frame_idx, frame_idx)
        mock_session.iterate_frames.return_value = iter(frames)

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            return_value=make_test_face_metrics(),
        ):
            result = ranker.rank_frames(mock_session, start_time, end_time)

        assert len(result.ranked_frames) == 1
        assert result.ranked_frames[0].rank == FIRST_RANK

    def test_no_face_detected_in_any_frame_still_returns_results(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 4.0
        end_time = 4.3
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)
        mock_session.iterate_frames.return_value = iter(frames)

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            return_value=make_test_face_metrics(detected=False),
        ):
            result = ranker.rank_frames(mock_session, start_time, end_time)

        assert len(result.ranked_frames) == len(frames)
        for rf in result.ranked_frames:
            assert rf.score == pytest.approx(0.0)

    def test_mixed_face_detection_scores_detected_frames_higher(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 6.0
        end_time = 6.2
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)
        mock_session.iterate_frames.return_value = iter(frames)

        detection_pattern = [True, False, True, False, True, False, True]

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            side_effect=[
                make_test_face_metrics(detected=detection_pattern[i % len(detection_pattern)])
                for i in range(len(frames))
            ],
        ):
            result = ranker.rank_frames(mock_session, start_time, end_time)

        top_half = result.ranked_frames[: len(result.ranked_frames) // 2]
        for rf in top_half:
            assert rf.score > 0.0


class TestInputValidation:
    def test_sample_rate_zero_raises_error(self, ranker, mock_session):
        with pytest.raises((ValueError, ZeroDivisionError)):
            mock_session.iterate_frames.side_effect = ZeroDivisionError()
            ranker.rank_frames(mock_session, 5.0, 6.0, sample_rate=0)

    def test_negative_sample_rate_raises_error(self, ranker, mock_session):
        mock_session.iterate_frames.return_value = iter([])

        with pytest.raises((ValueError, NoValidFramesError)):
            ranker.rank_frames(mock_session, 5.0, 6.0, sample_rate=-1)


class TestAsyncParity:
    def test_async_rank_frames_matches_sync_results(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 5.0
        end_time = 5.5
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)

        def reset_iterator(*_args, **_kwargs):
            return iter(frames)

        async def async_iterator(*_args, **_kwargs):
            for item in frames:
                yield item

        mock_session.iterate_frames.side_effect = reset_iterator
        mock_session.iterate_frames_async = async_iterator

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            return_value=make_test_face_metrics(),
        ):
            sync_result = ranker.rank_frames(mock_session, start_time, end_time)
            async_result = asyncio.run(ranker.rank_frames_async(mock_session, start_time, end_time))

        assert len(sync_result.ranked_frames) == len(async_result.ranked_frames)

        for sync_rf, async_rf in zip(sync_result.ranked_frames, async_result.ranked_frames):
            assert sync_rf.rank == async_rf.rank
            assert sync_rf.frame_index == async_rf.frame_index
            assert sync_rf.timestamp == pytest.approx(async_rf.timestamp)
            assert sync_rf.score == pytest.approx(async_rf.score)


class TestGetDetailedScores:
    def test_returns_frame_scores_sorted_by_final_score_descending(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 7.0
        end_time = 7.3
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)
        mock_session.iterate_frames.return_value = iter(frames)

        varying_ears = [VALID_EAR, 0.18, VALID_EAR + 0.02, 0.12]

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            side_effect=[
                make_test_face_metrics(ear=varying_ears[i % len(varying_ears)])
                for i in range(len(frames))
            ],
        ):
            scores = ranker.get_detailed_scores(mock_session, start_time, end_time)

        assert all(isinstance(s, FrameScore) for s in scores)
        final_scores = [s.final_score for s in scores]
        assert final_scores == sorted(final_scores, reverse=True)

    def test_raises_no_valid_frames_error_when_empty(self, ranker, mock_session):
        mock_session.iterate_frames.return_value = iter([])

        with pytest.raises(NoValidFramesError):
            ranker.get_detailed_scores(mock_session, 5.0, 6.0)

    def test_detailed_scores_contain_component_scores(
        self, ranker, mock_session, make_test_face_metrics
    ):
        start_time = 8.0
        end_time = 8.2
        start_frame = int(start_time * DEFAULT_FPS)
        end_frame = int(end_time * DEFAULT_FPS)

        frames = generate_dummy_frames(start_frame, end_frame)
        mock_session.iterate_frames.return_value = iter(frames)

        with patch.object(
            ranker.extractor,
            "extract_face_metrics",
            return_value=make_test_face_metrics(),
        ):
            scores = ranker.get_detailed_scores(mock_session, start_time, end_time)

        for score in scores:
            assert hasattr(score, "eye_score")
            assert hasattr(score, "mouth_score")
            assert hasattr(score, "final_score")
            assert score.eye_score >= 0.0
            assert score.mouth_score >= 0.0
