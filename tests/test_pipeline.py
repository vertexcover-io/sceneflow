"""Integration tests for the full pipeline with mocked external dependencies."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from sceneflow.api.public import get_cut_frames, cut_video
from sceneflow.shared.models import RankingResult
from sceneflow.shared.exceptions import (
    NoValidFramesError,
    VideoNotFoundError,
    VideoDownloadError,
    VideoOpenError,
    FFmpegNotFoundError,
)

from tests.helpers import (
    TEST_VIDEO_PATH,
    TEST_VIDEO_URL,
    TEST_OUTPUT_PATH,
    TEST_SPEECH_END_TIME,
    TEST_BEST_TIMESTAMP,
)


def setup_pipeline_mocks(
    mock_session_class, mock_detector_class, mock_ranker_class, make_video_session, ranked_frames
):
    mock_session = make_video_session()
    mock_session_class.return_value = mock_session

    mock_detector = Mock()
    mock_detector.get_speech_end_time_async = AsyncMock(return_value=(TEST_SPEECH_END_TIME, 0.9))
    mock_detector_class.return_value = mock_detector

    mock_ranker = Mock()
    mock_ranker.rank_frames_async = AsyncMock(
        return_value=RankingResult(ranked_frames=ranked_frames)
    )
    mock_ranker_class.return_value = mock_ranker

    return mock_session, mock_detector, mock_ranker


class TestGetCutFramesInputValidation:
    def test_n_less_than_one_raises_value_error(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            get_cut_frames(TEST_VIDEO_PATH, n=0)

    def test_n_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            get_cut_frames(TEST_VIDEO_PATH, n=-1)

    @patch("sceneflow.api.public.VideoSession")
    def test_nonexistent_file_raises_error(self, mock_session_class):
        mock_session_class.side_effect = VideoNotFoundError("/nonexistent/video.mp4")

        with pytest.raises(VideoNotFoundError):
            get_cut_frames("/nonexistent/video.mp4")


@patch("sceneflow.api.public.CutPointRanker")
@patch("sceneflow.api.public.SpeechDetector")
@patch("sceneflow.api.public.VideoSession")
class TestGetCutFramesCorePipeline:
    def test_local_video_returns_correct_number_of_timestamps(
        self,
        mock_session_class,
        mock_detector_class,
        mock_ranker_class,
        make_video_session,
        make_ranked_frames,
    ):
        ranked_frames = make_ranked_frames(count=5)
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames,
        )

        result = get_cut_frames(TEST_VIDEO_PATH, n=3)

        assert len(result) == 3
        assert result[0] == TEST_BEST_TIMESTAMP

    @patch("sceneflow.api.public.cleanup_downloaded_video_async", new_callable=AsyncMock)
    @patch("sceneflow.api.public.download_video_async", new_callable=AsyncMock)
    @patch("sceneflow.api.public.is_url", return_value=True)
    def test_url_video_downloads_processes_and_cleans_up(
        self,
        _mock_is_url,
        mock_download,
        mock_cleanup,
        mock_session_class,
        mock_detector_class,
        mock_ranker_class,
        make_video_session,
        make_ranked_frames,
    ):
        downloaded_path = "/tmp/sceneflow_test/video.mp4"
        mock_download.return_value = downloaded_path
        ranked_frames = make_ranked_frames(count=3)
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames,
        )

        result = get_cut_frames(TEST_VIDEO_URL, n=1)

        mock_download.assert_called_once_with(TEST_VIDEO_URL)
        mock_cleanup.assert_called_once_with(downloaded_path)
        assert len(result) == 1


@patch("sceneflow.api.public.CutPointRanker")
@patch("sceneflow.api.public.SpeechDetector")
@patch("sceneflow.api.public.VideoSession")
class TestGetCutFramesErrorHandling:
    def test_no_valid_frames_raises_error(
        self, mock_session_class, mock_detector_class, mock_ranker_class, make_video_session
    ):
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames=[],
        )

        with pytest.raises(NoValidFramesError):
            get_cut_frames(TEST_VIDEO_PATH)

    def test_corrupt_video_raises_error(
        self, mock_session_class, _mock_detector_class, _mock_ranker_class
    ):
        mock_session_class.side_effect = VideoOpenError(TEST_VIDEO_PATH, "corrupt file")

        with pytest.raises(VideoOpenError):
            get_cut_frames(TEST_VIDEO_PATH)

    @patch("sceneflow.api.public.download_video_async", new_callable=AsyncMock)
    @patch("sceneflow.api.public.is_url", return_value=True)
    def test_url_download_failure_raises_error(
        self,
        _mock_is_url,
        mock_download,
        _mock_session_class,
        _mock_detector_class,
        _mock_ranker_class,
    ):
        mock_download.side_effect = VideoDownloadError(TEST_VIDEO_URL, "HTTP error 404")

        with pytest.raises(VideoDownloadError):
            get_cut_frames(TEST_VIDEO_URL)


@patch("sceneflow.api.public.CutPointRanker")
@patch("sceneflow.api.public.SpeechDetector")
@patch("sceneflow.api.public.VideoSession")
class TestGetCutFramesResourceCleanup:
    @patch("sceneflow.api.public.cleanup_downloaded_video_async", new_callable=AsyncMock)
    @patch("sceneflow.api.public.download_video_async", new_callable=AsyncMock)
    @patch("sceneflow.api.public.is_url", return_value=True)
    def test_downloaded_video_cleaned_up_on_success(
        self,
        _mock_is_url,
        mock_download,
        mock_cleanup,
        mock_session_class,
        mock_detector_class,
        mock_ranker_class,
        make_video_session,
        make_ranked_frames,
    ):
        downloaded_path = "/tmp/sceneflow_test/video.mp4"
        mock_download.return_value = downloaded_path
        ranked_frames = make_ranked_frames(count=1)
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames,
        )

        get_cut_frames(TEST_VIDEO_URL)

        mock_cleanup.assert_called_once_with(downloaded_path)

    @patch("sceneflow.api.public.cleanup_downloaded_video_async", new_callable=AsyncMock)
    @patch("sceneflow.api.public.download_video_async", new_callable=AsyncMock)
    @patch("sceneflow.api.public.is_url", return_value=True)
    def test_downloaded_video_cleaned_up_on_exception(
        self,
        _mock_is_url,
        mock_download,
        mock_cleanup,
        mock_session_class,
        _mock_detector_class,
        _mock_ranker_class,
        make_video_session,
    ):
        downloaded_path = "/tmp/sceneflow_test/video.mp4"
        mock_download.return_value = downloaded_path
        mock_session_class.side_effect = VideoOpenError(downloaded_path, "corrupt file")

        with pytest.raises(VideoOpenError):
            get_cut_frames(TEST_VIDEO_URL)

        mock_cleanup.assert_called_once_with(downloaded_path)


@patch("sceneflow.api.public.LLMFrameSelector")
@patch("sceneflow.api.public.CutPointRanker")
@patch("sceneflow.api.public.SpeechDetector")
@patch("sceneflow.api.public.VideoSession")
class TestGetCutFramesLLMSelection:
    def test_llm_selection_used_when_n_equals_one(
        self,
        mock_session_class,
        mock_detector_class,
        mock_ranker_class,
        mock_selector_class,
        make_video_session,
        make_ranked_frames,
    ):
        ranked_frames = make_ranked_frames(count=5)
        llm_selected_frame = ranked_frames[2]
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames,
        )

        mock_selector = Mock()
        mock_selector.select_best_frame_async = AsyncMock(return_value=llm_selected_frame)
        mock_selector_class.return_value = mock_selector

        result = get_cut_frames(TEST_VIDEO_PATH, n=1, use_llm_selection=True)

        mock_selector.select_best_frame_async.assert_called_once()
        assert result[0] == llm_selected_frame.timestamp

    def test_llm_selection_not_used_when_n_greater_than_one(
        self,
        mock_session_class,
        mock_detector_class,
        mock_ranker_class,
        mock_selector_class,
        make_video_session,
        make_ranked_frames,
    ):
        ranked_frames = make_ranked_frames(count=5)
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames,
        )

        result = get_cut_frames(TEST_VIDEO_PATH, n=3, use_llm_selection=True)

        mock_selector_class.assert_not_called()
        assert len(result) == 3

    def test_llm_selection_skipped_when_only_one_frame(
        self,
        mock_session_class,
        mock_detector_class,
        mock_ranker_class,
        mock_selector_class,
        make_video_session,
        make_ranked_frames,
    ):
        ranked_frames = make_ranked_frames(count=1)
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames,
        )

        result = get_cut_frames(TEST_VIDEO_PATH, n=1, use_llm_selection=True)

        mock_selector_class.assert_not_called()
        assert len(result) == 1


@patch("sceneflow.api.public.CutPointRanker")
@patch("sceneflow.api.public.SpeechDetector")
@patch("sceneflow.api.public.VideoSession")
class TestGetCutFramesSpecialModes:
    def test_disable_visual_analysis_returns_speech_end_time(
        self, mock_session_class, mock_detector_class, mock_ranker_class, make_video_session
    ):
        mock_session = make_video_session()
        mock_session_class.return_value = mock_session

        mock_detector = Mock()
        mock_detector.get_speech_end_time_async = AsyncMock(
            return_value=(TEST_SPEECH_END_TIME, 0.9)
        )
        mock_detector_class.return_value = mock_detector

        result = get_cut_frames(TEST_VIDEO_PATH, disable_visual_analysis=True)

        mock_ranker_class.assert_not_called()
        assert result[0] == TEST_SPEECH_END_TIME


@patch("sceneflow.api.public._cut_video_util_async", new_callable=AsyncMock)
@patch("sceneflow.api.public.CutPointRanker")
@patch("sceneflow.api.public.SpeechDetector")
@patch("sceneflow.api.public.VideoSession")
class TestCutVideoCorePipeline:
    def test_cut_video_creates_output_and_returns_timestamp(
        self,
        mock_session_class,
        mock_detector_class,
        mock_ranker_class,
        mock_cut,
        make_video_session,
        make_ranked_frames,
    ):
        ranked_frames = make_ranked_frames(count=3)
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames,
        )

        result = cut_video(TEST_VIDEO_PATH, TEST_OUTPUT_PATH)

        mock_cut.assert_called_once_with(TEST_VIDEO_PATH, TEST_BEST_TIMESTAMP, TEST_OUTPUT_PATH)
        assert result == TEST_BEST_TIMESTAMP


@patch("sceneflow.api.public._cut_video_util_async", new_callable=AsyncMock)
@patch("sceneflow.api.public.CutPointRanker")
@patch("sceneflow.api.public.SpeechDetector")
@patch("sceneflow.api.public.VideoSession")
class TestCutVideoFFmpegIntegration:
    def test_ffmpeg_not_found_raises_error(
        self,
        mock_session_class,
        mock_detector_class,
        mock_ranker_class,
        mock_cut,
        make_video_session,
        make_ranked_frames,
    ):
        ranked_frames = make_ranked_frames(count=1)
        setup_pipeline_mocks(
            mock_session_class,
            mock_detector_class,
            mock_ranker_class,
            make_video_session,
            ranked_frames,
        )
        mock_cut.side_effect = FFmpegNotFoundError()

        with pytest.raises(FFmpegNotFoundError):
            cut_video(TEST_VIDEO_PATH, TEST_OUTPUT_PATH)


@patch("sceneflow.api.public._cut_video_util_async", new_callable=AsyncMock)
@patch("sceneflow.api.public.SpeechDetector")
@patch("sceneflow.api.public.VideoSession")
class TestCutVideoSaveOptions:
    @patch("sceneflow.api.public.save_analysis_logs")
    def test_save_logs_with_disable_visual_analysis_does_not_crash(
        self, mock_save_logs, mock_session_class, mock_detector_class, _mock_cut, make_video_session
    ):
        mock_session = make_video_session()
        mock_session_class.return_value = mock_session

        mock_detector = Mock()
        mock_detector.get_speech_end_time_async = AsyncMock(
            return_value=(TEST_SPEECH_END_TIME, 0.9)
        )
        mock_detector_class.return_value = mock_detector

        result = cut_video(
            TEST_VIDEO_PATH,
            TEST_OUTPUT_PATH,
            save_logs=True,
            disable_visual_analysis=True,
        )

        mock_save_logs.assert_not_called()
        assert result == TEST_SPEECH_END_TIME
