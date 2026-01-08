"""Tests for face metric calculations (EAR and MAR)."""

import pytest
import numpy as np

from sceneflow.extraction.face_metrics import euclidean_distance, calculate_ear, calculate_mar
from sceneflow.shared.constants import EAR, MAR


class TestEuclideanDistance:
    def test_same_point_returns_zero(self):
        p = np.array([5.0, 5.0])
        assert euclidean_distance(p, p) == 0.0

    def test_horizontal_distance(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 0.0])
        assert euclidean_distance(p1, p2) == 3.0

    def test_vertical_distance(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.0, 4.0])
        assert euclidean_distance(p1, p2) == 4.0

    def test_diagonal_distance(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        assert euclidean_distance(p1, p2) == 5.0

    def test_handles_negative_coordinates(self):
        p1 = np.array([-3.0, -4.0])
        p2 = np.array([0.0, 0.0])
        assert euclidean_distance(p1, p2) == 5.0


class TestCalculateEAR:
    def test_normal_open_eye(self, normal_eye_landmarks):
        ear = calculate_ear(normal_eye_landmarks)
        assert EAR.MIN_VALID <= ear <= EAR.MAX_VALID
        assert EAR.NORMAL_MIN <= ear <= EAR.NORMAL_MAX

    def test_closed_eye_returns_default(self, closed_eye_landmarks):
        ear = calculate_ear(closed_eye_landmarks)
        assert ear == EAR.DEFAULT or ear < EAR.NORMAL_MIN

    def test_empty_landmarks_returns_default(self):
        ear = calculate_ear(np.array([]))
        assert ear == EAR.DEFAULT

    def test_insufficient_landmarks_returns_default(self):
        landmarks = np.array([[0, 0], [1, 1], [2, 2]])
        ear = calculate_ear(landmarks)
        assert ear == EAR.DEFAULT

    def test_zero_horizontal_distance_returns_default(self):
        landmarks = np.array(
            [[10, 10], [10, 5], [10, 4], [10, 5], [10, 10], [10, 16]], dtype=np.float64
        )
        ear = calculate_ear(landmarks)
        assert ear == EAR.DEFAULT

    def test_ear_outside_valid_range_returns_default(self):
        landmarks = np.array(
            [[0, 10], [5, 0], [10, 0], [15, 0], [20, 10], [10, 30]], dtype=np.float64
        )
        ear = calculate_ear(landmarks)
        assert ear == EAR.DEFAULT


class TestCalculateMAR:
    def test_closed_mouth(self, closed_mouth_landmarks):
        mar = calculate_mar(closed_mouth_landmarks)
        assert MAR.MIN_VALID <= mar <= MAR.MAX_VALID

    def test_open_mouth(self, open_mouth_landmarks):
        mar = calculate_mar(open_mouth_landmarks)
        assert MAR.MIN_VALID <= mar <= MAR.MAX_VALID

    def test_empty_landmarks_returns_default(self):
        mar = calculate_mar(np.array([]))
        assert mar == MAR.DEFAULT

    def test_insufficient_landmarks_returns_default(self):
        landmarks = np.array([[0, 0]] * 10)
        mar = calculate_mar(landmarks)
        assert mar == MAR.DEFAULT

    def test_zero_horizontal_distance_returns_default(self):
        landmarks = np.zeros((20, 2), dtype=np.float64)
        landmarks[0] = [25, 50]
        landmarks[6] = [25, 50]
        landmarks[2] = [25, 45]
        landmarks[4] = [25, 45]
        landmarks[8] = [25, 55]
        landmarks[10] = [25, 55]
        mar = calculate_mar(landmarks)
        assert mar == MAR.DEFAULT


class TestEARBoundaryValues:
    @pytest.mark.parametrize(
        "ear_value,expected_valid",
        [
            (0.07, False),
            (0.08, True),
            (0.25, True),
            (0.28, True),
            (0.32, True),
            (0.35, True),
            (0.36, False),
        ],
    )
    def test_ear_boundary_validation(self, ear_value, expected_valid):
        is_valid = EAR.MIN_VALID <= ear_value <= EAR.MAX_VALID
        assert is_valid == expected_valid


class TestMARBoundaryValues:
    @pytest.mark.parametrize(
        "mar_value,expected_valid",
        [
            (0.14, False),
            (0.15, True),
            (0.20, True),
            (0.27, True),
            (0.35, True),
            (1.50, True),
            (1.51, False),
        ],
    )
    def test_mar_boundary_validation(self, mar_value, expected_valid):
        is_valid = MAR.MIN_VALID <= mar_value <= MAR.MAX_VALID
        assert is_valid == expected_valid
