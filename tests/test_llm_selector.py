"""Tests for LLMFrameSelector with mocked OpenAI API."""

import os
import pytest
from unittest.mock import Mock, patch

from sceneflow.selection.llm_selector import LLMFrameSelector
from sceneflow.shared.constants import LLM

from tests.helpers import (
    TEST_API_KEY,
    TEST_SPEECH_END_TIME,
    DEFAULT_DURATION,
    TEST_FRAME_SCORE,
    TEST_JPEG_BYTES,
    make_ranked_frame,
    make_ranked_frames_list,
)


@pytest.fixture
def mock_session():
    session = Mock()
    session.get_frame_as_jpeg = Mock(return_value=TEST_JPEG_BYTES)
    return session


@pytest.fixture
def selector():
    with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_API_KEY}):
        with patch("sceneflow.selection.llm_selector.OpenAI"):
            return LLMFrameSelector()


class TestEmptyFramesRaisesError:
    def test_empty_frames_raises_value_error(self, selector, mock_session):
        with pytest.raises(ValueError, match="No frames provided for selection"):
            selector.select_best_frame(
                session=mock_session,
                ranked_frames=[],
                speech_end_time=TEST_SPEECH_END_TIME,
                video_duration=DEFAULT_DURATION,
            )


class TestSingleFrameShortCircuit:
    def test_single_frame_returns_directly(self, selector, mock_session):
        single_frame = make_ranked_frame(
            rank=1, frame_index=100, timestamp=8.5, score=TEST_FRAME_SCORE
        )

        result = selector.select_best_frame(
            session=mock_session,
            ranked_frames=[single_frame],
            speech_end_time=TEST_SPEECH_END_TIME,
            video_duration=DEFAULT_DURATION,
        )

        assert result == single_frame
        mock_session.get_frame_as_jpeg.assert_not_called()


@patch("sceneflow.selection.llm_selector.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": TEST_API_KEY})
class TestValidResponseReturnsCorrectFrame:
    def test_response_1_returns_first_frame(self, mock_openai, mock_session, mock_openai_response):
        mock_openai_response.choices[0].message.content = "1"
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client

        selector = LLMFrameSelector()
        frames = make_ranked_frames_list(count=3, start_timestamp=8.5)

        result = selector.select_best_frame(
            session=mock_session,
            ranked_frames=frames,
            speech_end_time=TEST_SPEECH_END_TIME,
            video_duration=DEFAULT_DURATION,
        )

        assert result == frames[0]
        assert result.rank == 1


@patch("sceneflow.selection.llm_selector.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": TEST_API_KEY})
class TestBoundaryResponseReturnsCorrectFrame:
    def test_response_at_boundary_returns_correct_frame(
        self, mock_openai, mock_session, mock_openai_response
    ):
        num_frames = LLM.TOP_CANDIDATES_COUNT
        mock_openai_response.choices[0].message.content = str(num_frames)

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client

        selector = LLMFrameSelector()
        frames = make_ranked_frames_list(count=num_frames, start_timestamp=8.5)

        result = selector.select_best_frame(
            session=mock_session,
            ranked_frames=frames,
            speech_end_time=TEST_SPEECH_END_TIME,
            video_duration=DEFAULT_DURATION,
        )

        assert result == frames[num_frames - 1]
        assert result.rank == num_frames


@patch("sceneflow.selection.llm_selector.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": TEST_API_KEY})
class TestOutOfRangeResponseFallback:
    @pytest.mark.parametrize("invalid_response", ["0", "6", "10", "-1", "99"])
    def test_out_of_range_falls_back_to_first(
        self, mock_openai, mock_session, mock_openai_response, invalid_response
    ):
        mock_openai_response.choices[0].message.content = invalid_response

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client

        selector = LLMFrameSelector()
        frames = make_ranked_frames_list(count=LLM.TOP_CANDIDATES_COUNT, start_timestamp=8.5)

        result = selector.select_best_frame(
            session=mock_session,
            ranked_frames=frames,
            speech_end_time=TEST_SPEECH_END_TIME,
            video_duration=DEFAULT_DURATION,
        )

        assert result == frames[0]


@patch("sceneflow.selection.llm_selector.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": TEST_API_KEY})
class TestNonNumericResponseFallback:
    @pytest.mark.parametrize(
        "non_numeric_response",
        ["Frame 3", "The best frame is 2", "three", "", "   ", "2.5", "1st", "None"],
    )
    def test_non_numeric_falls_back_to_first(
        self, mock_openai, mock_session, mock_openai_response, non_numeric_response
    ):
        mock_openai_response.choices[0].message.content = non_numeric_response

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client

        selector = LLMFrameSelector()
        frames = make_ranked_frames_list(count=3, start_timestamp=8.5)

        result = selector.select_best_frame(
            session=mock_session,
            ranked_frames=frames,
            speech_end_time=TEST_SPEECH_END_TIME,
            video_duration=DEFAULT_DURATION,
        )

        assert result == frames[0]


@patch("sceneflow.selection.llm_selector.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": TEST_API_KEY})
class TestAllExtractionFailuresRaisesError:
    def test_all_extraction_failures_raises_value_error(self, mock_openai, mock_openai_response):
        failing_session = Mock()
        failing_session.get_frame_as_jpeg.side_effect = Exception("Frame extraction failed")

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client

        selector = LLMFrameSelector()
        frames = make_ranked_frames_list(count=3, start_timestamp=8.5)

        with pytest.raises(ValueError, match="No valid frame data for LLM selection"):
            selector.select_best_frame(
                session=failing_session,
                ranked_frames=frames,
                speech_end_time=TEST_SPEECH_END_TIME,
                video_duration=DEFAULT_DURATION,
            )


@patch("sceneflow.selection.llm_selector.OpenAI")
@patch.dict(os.environ, {"OPENAI_API_KEY": TEST_API_KEY})
class TestPartialExtractionFailuresContinues:
    def test_partial_failures_uses_remaining_frames(self, mock_openai, mock_openai_response):
        partial_session = Mock()
        partial_session.get_frame_as_jpeg.side_effect = [
            Exception("First frame failed"),
            TEST_JPEG_BYTES,
            TEST_JPEG_BYTES,
        ]

        mock_openai_response.choices[0].message.content = "1"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client

        selector = LLMFrameSelector()
        frames = make_ranked_frames_list(count=3, start_timestamp=8.5)

        result = selector.select_best_frame(
            session=partial_session,
            ranked_frames=frames,
            speech_end_time=TEST_SPEECH_END_TIME,
            video_duration=DEFAULT_DURATION,
        )

        assert result == frames[1]


class TestMissingApiKeyRaisesError:
    def test_no_api_key_raises_value_error(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)

            with pytest.raises(ValueError, match="OpenAI API key not found"):
                LLMFrameSelector()

    @patch("sceneflow.selection.llm_selector.OpenAI")
    def test_api_key_from_parameter_works(self, _mock_openai):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)

            selector = LLMFrameSelector(api_key=TEST_API_KEY)
            assert selector.api_key == TEST_API_KEY

    @patch("sceneflow.selection.llm_selector.OpenAI")
    def test_api_key_from_env_var_works(self, _mock_openai):
        with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_API_KEY}):
            selector = LLMFrameSelector()
            assert selector.api_key == TEST_API_KEY
