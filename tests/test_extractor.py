"""Tests for FeatureExtractor with mocked InsightFace."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from sceneflow.extraction.extractor import FeatureExtractor
from sceneflow.shared.constants import EAR, MAR, INSIGHTFACE
from sceneflow.shared.models import FaceMetrics

from tests.helpers import (
    SMALL_WIDTH,
    SMALL_HEIGHT,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    TOTAL_LANDMARKS_REQUIRED,
    make_valid_landmarks,
)


VALID_BBOX = np.array([100.0, 100.0, 300.0, 300.0])
SMALL_BBOX = np.array([150.0, 150.0, 200.0, 200.0])
EDGE_BBOX = np.array([0.0, 0.0, 50.0, 50.0])

HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.3
THRESHOLD_CONFIDENCE = INSIGHTFACE.MIN_FACE_CONFIDENCE
EXPECTED_CENTER_X = (VALID_BBOX[0] + VALID_BBOX[2]) / 2
EXPECTED_CENTER_Y = (VALID_BBOX[1] + VALID_BBOX[3]) / 2

EXPECTED_EAR = 0.3
EXPECTED_MAR = 0.217
METRIC_TOLERANCE = 0.01


@pytest.fixture
def mock_insightface():
    with patch("sceneflow.extraction.extractor.FaceAnalysis") as mock_class:
        mock_app = MagicMock()
        mock_class.return_value = mock_app
        yield mock_app


@pytest.fixture
def extractor(mock_insightface):
    return FeatureExtractor()


@pytest.fixture
def test_frame():
    return np.zeros((SMALL_HEIGHT, SMALL_WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def make_test_mock_face():
    def _make(
        bbox: np.ndarray = None,
        det_score: float = 0.9,
        landmarks: np.ndarray = None,
        has_det_score: bool = True,
        has_landmarks: bool = True,
    ):
        from unittest.mock import Mock

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
                landmarks = make_valid_landmarks()
            face.landmark_2d_106 = landmarks
        else:
            face.landmark_2d_106 = None

        return face

    return _make


class TestExtractFaceMetricsHappyPath:
    def test_returns_face_metrics_with_detected_true(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face()
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert isinstance(result, FaceMetrics)
        assert result.detected is True

    def test_returns_expected_ear_value(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face()
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert abs(result.ear - EXPECTED_EAR) < METRIC_TOLERANCE

    def test_returns_expected_mar_value(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face()
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert abs(result.mar - EXPECTED_MAR) < METRIC_TOLERANCE

    def test_returns_correct_face_center(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face(bbox=VALID_BBOX)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.center == (EXPECTED_CENTER_X, EXPECTED_CENTER_Y)

    def test_returns_positive_sharpness(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face()
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.sharpness >= 0.0


class TestNoFaceDetection:
    def test_empty_faces_list_returns_detected_false(self, extractor, mock_insightface, test_frame):
        mock_insightface.get.return_value = []

        result = extractor.extract_face_metrics(test_frame)

        assert result.detected is False

    def test_empty_faces_returns_default_ear(self, extractor, mock_insightface, test_frame):
        mock_insightface.get.return_value = []

        result = extractor.extract_face_metrics(test_frame)

        assert result.ear == EAR.DEFAULT

    def test_empty_faces_returns_default_mar(self, extractor, mock_insightface, test_frame):
        mock_insightface.get.return_value = []

        result = extractor.extract_face_metrics(test_frame)

        assert result.mar == MAR.DEFAULT

    def test_empty_faces_returns_zero_sharpness(self, extractor, mock_insightface, test_frame):
        mock_insightface.get.return_value = []

        result = extractor.extract_face_metrics(test_frame)

        assert result.sharpness == 0.0

    def test_empty_faces_returns_zero_center(self, extractor, mock_insightface, test_frame):
        mock_insightface.get.return_value = []

        result = extractor.extract_face_metrics(test_frame)

        assert result.center == (0.0, 0.0)


class TestConfidenceThresholdFiltering:
    def test_low_confidence_face_filtered_out(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        low_conf_face = make_test_mock_face(det_score=LOW_CONFIDENCE)
        mock_insightface.get.return_value = [low_conf_face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.detected is False

    def test_face_at_exact_threshold_included(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        threshold_face = make_test_mock_face(det_score=THRESHOLD_CONFIDENCE)
        mock_insightface.get.return_value = [threshold_face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.detected is True

    def test_mixed_confidence_uses_high_confidence_face(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        low_conf_large = make_test_mock_face(bbox=VALID_BBOX, det_score=LOW_CONFIDENCE)
        high_conf_small = make_test_mock_face(bbox=SMALL_BBOX, det_score=HIGH_CONFIDENCE)
        mock_insightface.get.return_value = [low_conf_large, high_conf_small]

        result = extractor.extract_face_metrics(test_frame)

        expected_center = (
            (SMALL_BBOX[0] + SMALL_BBOX[2]) / 2,
            (SMALL_BBOX[1] + SMALL_BBOX[3]) / 2,
        )
        assert result.center == expected_center

    def test_face_without_det_score_attribute_filtered(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face_no_score = make_test_mock_face(has_det_score=False)
        mock_insightface.get.return_value = [face_no_score]

        result = extractor.extract_face_metrics(test_frame)

        assert result.detected is False


class TestMissingLandmarks:
    def test_none_landmarks_returns_default_ear(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face(has_landmarks=False)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.ear == EAR.DEFAULT

    def test_none_landmarks_returns_default_mar(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face(has_landmarks=False)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.mar == MAR.DEFAULT

    def test_none_landmarks_still_returns_detected_true(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face(has_landmarks=False)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.detected is True


class TestOneEyeFallback:
    def test_invalid_left_eye_uses_right_ear(
        self, extractor, mock_insightface, test_frame, invalid_eye_landmarks, make_test_mock_face
    ):
        face = make_test_mock_face(landmarks=invalid_eye_landmarks)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert EAR.MIN_VALID <= result.ear <= EAR.MAX_VALID

    def test_invalid_right_eye_uses_left_ear(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
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

        collapsed_point = [180, 110]
        for idx in RIGHT_EYE_INDICES:
            landmarks[idx] = collapsed_point

        face = make_test_mock_face(landmarks=landmarks)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert EAR.MIN_VALID <= result.ear <= EAR.MAX_VALID

    def test_both_eyes_invalid_returns_default(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        landmarks = make_valid_landmarks()
        for idx in LEFT_EYE_INDICES:
            landmarks[idx] = [100, 110]
        for idx in RIGHT_EYE_INDICES:
            landmarks[idx] = [180, 110]

        face = make_test_mock_face(landmarks=landmarks)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.ear == EAR.DEFAULT


class TestEmptyFaceRegionSharpness:
    def test_zero_area_bbox_returns_zero_sharpness(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        zero_area_bbox = np.array([100.0, 100.0, 100.0, 200.0])
        face = make_test_mock_face(bbox=zero_area_bbox)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.sharpness == 0.0

    def test_inverted_bbox_returns_zero_sharpness(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        inverted_bbox = np.array([200.0, 200.0, 100.0, 100.0])
        face = make_test_mock_face(bbox=inverted_bbox)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.sharpness == 0.0


class TestBoundingBoxEdgeClamping:
    def test_bbox_at_origin_clamps_without_error(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        face = make_test_mock_face(bbox=EDGE_BBOX)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.detected is True
        assert result.sharpness >= 0.0

    def test_bbox_at_frame_edge_clamps_without_error(
        self, extractor, mock_insightface, test_frame, make_test_mock_face
    ):
        edge_bbox = np.array(
            [
                SMALL_WIDTH - 50,
                SMALL_HEIGHT - 50,
                SMALL_WIDTH,
                SMALL_HEIGHT,
            ],
            dtype=np.float32,
        )
        face = make_test_mock_face(bbox=edge_bbox)
        mock_insightface.get.return_value = [face]

        result = extractor.extract_face_metrics(test_frame)

        assert result.detected is True
