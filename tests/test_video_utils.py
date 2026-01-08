"""Tests for video utility functions."""

import subprocess
import cv2
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import requests

from sceneflow.utils.video import (
    is_url,
    validate_video_url,
    download_video,
    cleanup_downloaded_video,
    cut_video,
    cut_video_to_bytes,
    VideoCapture,
    VideoSession,
)
from sceneflow.shared.exceptions import (
    VideoDownloadError,
    InvalidURLError,
    FFmpegNotFoundError,
    FFmpegExecutionError,
)
from sceneflow.shared.constants import VIDEO

from tests.helpers import (
    DEFAULT_FPS,
    DEFAULT_FRAME_COUNT,
    SMALL_WIDTH,
    SMALL_HEIGHT,
)


TEST_HTTPS_URL = "https://example.com/video.mp4"
TEST_HTTP_URL = "http://example.com/video.mp4"
TEST_LOCAL_PATH = "/path/to/video.mp4"

TEST_DURATION = DEFAULT_FRAME_COUNT / DEFAULT_FPS
TEST_FRAME_INDEX_OUT_OF_RANGE = 999
TEST_CUT_TIMESTAMP = 5.0
TEST_VIDEO_BYTES = b"fake video content"
SCENEFLOW_TEMP_DIR_PREFIX = "sceneflow_"


class TestValidateVideoURL:
    def test_raises_on_invalid_scheme(self):
        with pytest.raises(InvalidURLError):
            validate_video_url("ftp://example.com/video.mp4")

    def test_raises_on_missing_host(self):
        with pytest.raises(InvalidURLError):
            validate_video_url("http:///video.mp4")


@patch("sceneflow.utils.video.requests.get")
class TestDownloadVideo:
    def test_downloads_to_temp_file(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"fake video data"]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = download_video("https://example.com/video.mp4")

        assert result.endswith(".mp4")

    def test_raises_on_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()

        with pytest.raises(VideoDownloadError, match="timed out"):
            download_video("https://example.com/video.mp4")

    def test_raises_on_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()

        with pytest.raises(VideoDownloadError, match="Connection failed"):
            download_video(TEST_HTTPS_URL)

    def test_raises_on_http_error(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(VideoDownloadError, match="HTTP error"):
            download_video(TEST_HTTPS_URL)


@patch("sceneflow.utils.video.subprocess.run")
class TestCutVideo:
    def test_cuts_video_successfully(self, mock_run, tmp_path):
        mock_run.return_value = Mock(returncode=0, stderr="")

        input_video = tmp_path / "input.mp4"
        input_video.touch()
        output_video = tmp_path / "output.mp4"

        result = cut_video(str(input_video), 5.0, str(output_video))

        mock_run.assert_called_once()
        assert result == str(output_video)

    def test_raises_ffmpeg_not_found(self, mock_run, tmp_path):
        mock_run.side_effect = FileNotFoundError()
        input_video = tmp_path / "input.mp4"
        input_video.touch()

        with pytest.raises(FFmpegNotFoundError):
            cut_video(str(input_video), 5.0, str(tmp_path / "output.mp4"))

    def test_raises_on_ffmpeg_error(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="ffmpeg", stderr="FFmpeg error"
        )

        input_video = tmp_path / "input.mp4"
        input_video.touch()

        with pytest.raises(FFmpegExecutionError):
            cut_video(str(input_video), 5.0, str(tmp_path / "output.mp4"))


class TestIsUrl:
    def test_returns_true_for_https_url(self):
        assert is_url(TEST_HTTPS_URL) is True

    def test_returns_true_for_http_url(self):
        assert is_url(TEST_HTTP_URL) is True

    def test_returns_false_for_local_path(self):
        assert is_url(TEST_LOCAL_PATH) is False

    def test_returns_false_for_empty_string(self):
        assert is_url("") is False


class TestCleanupDownloadedVideo:
    def test_deletes_existing_video_file(self, tmp_path):
        video_file = tmp_path / "video.mp4"
        video_file.touch()

        cleanup_downloaded_video(str(video_file))

        assert not video_file.exists()

    def test_deletes_empty_sceneflow_parent_directory(self, tmp_path):
        temp_dir = tmp_path / f"{SCENEFLOW_TEMP_DIR_PREFIX}test123"
        temp_dir.mkdir()
        video_file = temp_dir / "video.mp4"
        video_file.touch()

        cleanup_downloaded_video(str(video_file))

        assert not video_file.exists()
        assert not temp_dir.exists()


@patch("sceneflow.utils.video.cv2.VideoCapture")
class TestVideoCapture:
    def test_returns_open_capture_on_enter(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2_capture.return_value = mock_cap

        with VideoCapture(TEST_LOCAL_PATH) as cap:
            assert cap is mock_cap
            mock_cv2_capture.assert_called_once_with(TEST_LOCAL_PATH)

    def test_releases_capture_on_normal_exit(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2_capture.return_value = mock_cap

        with VideoCapture(TEST_LOCAL_PATH):
            pass

        mock_cap.release.assert_called_once()

    def test_releases_capture_on_exception_exit(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2_capture.return_value = mock_cap

        with pytest.raises(RuntimeError):
            with VideoCapture(TEST_LOCAL_PATH):
                raise RuntimeError("Test exception")

        mock_cap.release.assert_called_once()


@patch("sceneflow.utils.video.cv2.VideoCapture")
class TestVideoSession:
    def _create_mock_capture(self, mock_cv2_capture, num_frames=DEFAULT_FRAME_COUNT):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: DEFAULT_FPS,
            cv2.CAP_PROP_FRAME_COUNT: num_frames,
            cv2.CAP_PROP_FRAME_WIDTH: SMALL_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: SMALL_HEIGHT,
        }.get(prop, 0)

        test_frame = np.zeros((SMALL_HEIGHT, SMALL_WIDTH, 3), dtype=np.uint8)
        read_results = [(True, test_frame.copy()) for _ in range(num_frames)] + [(False, None)]
        mock_cap.read.side_effect = read_results
        mock_cv2_capture.return_value = mock_cap
        return mock_cap, test_frame

    def test_properties_returns_correct_video_properties(self, mock_cv2_capture):
        num_frames = DEFAULT_FRAME_COUNT
        self._create_mock_capture(mock_cv2_capture, num_frames=num_frames)

        with VideoSession(TEST_LOCAL_PATH) as session:
            props = session.properties
            assert props.fps == DEFAULT_FPS
            assert props.frame_count == num_frames
            assert props.width == SMALL_WIDTH
            assert props.height == SMALL_HEIGHT
            assert props.duration == num_frames / DEFAULT_FPS

    @patch("sceneflow.utils.video._extract_frame")
    def test_get_frame_returns_correct_frame(self, mock_extract_frame, mock_cv2_capture):
        num_frames = 3
        _, test_frame = self._create_mock_capture(mock_cv2_capture, num_frames=num_frames)
        mock_extract_frame.return_value = test_frame

        with VideoSession(TEST_LOCAL_PATH) as session:
            frame = session.get_frame(0)
            assert frame.shape == test_frame.shape
            mock_extract_frame.assert_called_once_with(TEST_LOCAL_PATH, 0)

    def test_get_frame_raises_index_error_for_out_of_range(self, mock_cv2_capture):
        self._create_mock_capture(mock_cv2_capture, num_frames=3)

        with VideoSession(TEST_LOCAL_PATH) as session:
            with pytest.raises(IndexError):
                session.get_frame(TEST_FRAME_INDEX_OUT_OF_RANGE)

    @patch("sceneflow.utils.video._extract_frame")
    @patch("sceneflow.utils.video.cv2.imencode")
    def test_get_frame_as_jpeg_returns_valid_bytes(
        self, mock_imencode, mock_extract_frame, mock_cv2_capture
    ):
        num_frames = 3
        _, test_frame = self._create_mock_capture(mock_cv2_capture, num_frames=num_frames)
        mock_extract_frame.return_value = test_frame
        expected_buffer = np.array([0xFF, 0xD8, 0xFF], dtype=np.uint8)
        mock_imencode.return_value = (True, expected_buffer)

        with VideoSession(TEST_LOCAL_PATH) as session:
            jpeg_bytes = session.get_frame_as_jpeg(0)
            assert jpeg_bytes == expected_buffer.tobytes()
            mock_imencode.assert_called_once()
            call_args = mock_imencode.call_args[0]
            assert call_args[0] == ".jpg"
            assert call_args[2] == [cv2.IMWRITE_JPEG_QUALITY, VIDEO.JPEG_QUALITY_DEFAULT]


class TestCutVideoToBytes:
    @patch("sceneflow.utils.video.subprocess.run")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("sceneflow.utils.video.tempfile.NamedTemporaryFile")
    def test_returns_video_bytes_on_success(self, mock_tempfile, mock_open, mock_run):
        mock_temp = MagicMock()
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)
        mock_temp.name = "/tmp/fake_temp.mp4"
        mock_tempfile.return_value = mock_temp

        mock_run.return_value = Mock(returncode=0)

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = TEST_VIDEO_BYTES
        mock_open.return_value = mock_file

        result = cut_video_to_bytes(TEST_LOCAL_PATH, TEST_CUT_TIMESTAMP)

        assert result == TEST_VIDEO_BYTES
        mock_run.assert_called_once()

    @patch("sceneflow.utils.video.subprocess.run")
    @patch("sceneflow.utils.video.tempfile.NamedTemporaryFile")
    def test_raises_ffmpeg_not_found(self, mock_tempfile, mock_run):
        mock_temp = MagicMock()
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)
        mock_temp.name = "/tmp/fake_temp.mp4"
        mock_tempfile.return_value = mock_temp

        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(FFmpegNotFoundError):
            cut_video_to_bytes(TEST_LOCAL_PATH, TEST_CUT_TIMESTAMP)

    @patch("sceneflow.utils.video.subprocess.run")
    @patch("sceneflow.utils.video.tempfile.NamedTemporaryFile")
    def test_raises_ffmpeg_execution_error_on_failure(self, mock_tempfile, mock_run):
        mock_temp = MagicMock()
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)
        mock_temp.name = "/tmp/fake_temp.mp4"
        mock_tempfile.return_value = mock_temp

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="ffmpeg", stderr="encoding failed"
        )

        with pytest.raises(FFmpegExecutionError):
            cut_video_to_bytes(TEST_LOCAL_PATH, TEST_CUT_TIMESTAMP)

    @patch("sceneflow.utils.video.subprocess.run")
    @patch("sceneflow.utils.video.tempfile.NamedTemporaryFile")
    def test_raises_ffmpeg_execution_error_on_timeout(self, mock_tempfile, mock_run):
        mock_temp = MagicMock()
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)
        mock_temp.name = "/tmp/fake_temp.mp4"
        mock_tempfile.return_value = mock_temp

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)

        with pytest.raises(FFmpegExecutionError, match="Timeout"):
            cut_video_to_bytes(TEST_LOCAL_PATH, TEST_CUT_TIMESTAMP)
