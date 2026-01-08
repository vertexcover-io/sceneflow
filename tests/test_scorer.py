"""Tests for frame scoring logic."""

import pytest

from sceneflow.core.scorer import FrameScorer
from sceneflow.shared.config import RankingConfig
from sceneflow.shared.constants import SMALL_MOVEMENT_TOLERANCE_PIXELS

from tests.helpers import (
    VALID_EAR,
    VALID_MAR,
    EAR_BELOW_VALID,
    EAR_ABOVE_VALID,
    EAR_SQUINTING,
    EAR_WIDE_OPEN,
    MAR_BELOW_VALID,
    MAR_ABOVE_VALID,
    MAR_SLIGHTLY_OPEN,
    MAR_TALKING,
    DEFAULT_SHARPNESS,
    BELOW_THRESHOLD_SHARPNESS,
    DEFAULT_TIMESTAMP,
    DEFAULT_FACE_CENTER,
)


@pytest.fixture
def scorer() -> FrameScorer:
    return FrameScorer(RankingConfig())


@pytest.fixture
def make_test_features():
    def _make(
        frame_index: int = 0,
        timestamp: float = DEFAULT_TIMESTAMP,
        eye_openness: float = None,
        mouth_openness: float = None,
        face_detected: bool = True,
        sharpness: float = DEFAULT_SHARPNESS,
        face_center: tuple = DEFAULT_FACE_CENTER,
    ):
        from sceneflow.shared.models import FrameFeatures

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


class TestComputeScoresCore:
    def test_empty_features_returns_empty_list(self, scorer):
        result = scorer.compute_scores([])
        assert result == []

    def test_single_frame_with_valid_face_returns_score(self, scorer, make_test_features):
        features = [make_test_features()]
        result = scorer.compute_scores(features)

        assert len(result) == 1
        assert result[0].frame_index == 0
        assert result[0].final_score > 0

    def test_face_not_detected_all_scores_zero(self, scorer, make_test_features):
        features = [make_test_features(face_detected=False)]
        result = scorer.compute_scores(features)

        assert len(result) == 1
        assert result[0].eye_score == 0.0
        assert result[0].mouth_score == 0.0
        assert result[0].final_score == 0.0


class TestEyeScoreBoundaries:
    def test_ear_in_normal_range_scores_one(self, scorer, make_test_features):
        features = [make_test_features(eye_openness=VALID_EAR)]
        result = scorer.compute_scores(features)
        assert result[0].eye_score == 1.0

    def test_ear_outside_valid_range_scores_zero(self, scorer, make_test_features):
        below_valid = [make_test_features(frame_index=0, eye_openness=EAR_BELOW_VALID)]
        above_valid = [make_test_features(frame_index=0, eye_openness=EAR_ABOVE_VALID)]

        result_below = scorer.compute_scores(below_valid)
        result_above = scorer.compute_scores(above_valid)

        assert result_below[0].eye_score == 0.0
        assert result_above[0].eye_score == 0.0

    def test_ear_valid_but_outside_normal_scores_zero(self, scorer, make_test_features):
        squinting = [make_test_features(frame_index=0, eye_openness=EAR_SQUINTING)]
        wide_open = [make_test_features(frame_index=0, eye_openness=EAR_WIDE_OPEN)]

        result_squinting = scorer.compute_scores(squinting)
        result_wide = scorer.compute_scores(wide_open)

        assert result_squinting[0].eye_score == 0.0
        assert result_wide[0].eye_score == 0.0


class TestMouthScoreBoundaries:
    def test_mar_in_closed_range_scores_one(self, scorer, make_test_features):
        features = [make_test_features(mouth_openness=VALID_MAR)]
        result = scorer.compute_scores(features)
        assert result[0].mouth_score == 1.0

    def test_mar_outside_valid_range_scores_zero(self, scorer, make_test_features):
        below_valid = [make_test_features(frame_index=0, mouth_openness=MAR_BELOW_VALID)]
        above_valid = [make_test_features(frame_index=0, mouth_openness=MAR_ABOVE_VALID)]

        result_below = scorer.compute_scores(below_valid)
        result_above = scorer.compute_scores(above_valid)

        assert result_below[0].mouth_score == 0.0
        assert result_above[0].mouth_score == 0.0

    def test_mar_valid_but_outside_closed_scores_zero(self, scorer, make_test_features):
        slightly_open = [make_test_features(frame_index=0, mouth_openness=MAR_SLIGHTLY_OPEN)]
        talking = [make_test_features(frame_index=0, mouth_openness=MAR_TALKING)]

        result_slightly = scorer.compute_scores(slightly_open)
        result_talking = scorer.compute_scores(talking)

        assert result_slightly[0].mouth_score == 0.0
        assert result_talking[0].mouth_score == 0.0


class TestConsistencyScoreCriticalPaths:
    def test_fewer_than_three_frames_all_consistency_scores_one(self, scorer, make_test_features):
        one_frame = [make_test_features(frame_index=0)]
        two_frames = [make_test_features(frame_index=0), make_test_features(frame_index=1)]

        result_one = scorer.compute_scores(one_frame)
        result_two = scorer.compute_scores(two_frames)

        assert result_one[0].motion_stability_score == 1.0
        assert result_two[0].motion_stability_score == 1.0
        assert result_two[1].motion_stability_score == 1.0

    def test_face_not_detected_consistency_score_zero(self, scorer, make_test_features):
        features = [
            make_test_features(frame_index=0),
            make_test_features(frame_index=1, face_detected=False),
            make_test_features(frame_index=2),
        ]
        result = scorer.compute_scores(features)
        assert result[1].motion_stability_score == 0.0

    def test_movement_within_tolerance_vs_exceeding_max(self, scorer, make_test_features):
        config = RankingConfig()
        small_movement = SMALL_MOVEMENT_TOLERANCE_PIXELS - 1.0
        large_movement = config.max_position_delta + 10.0

        stable_frames = [
            make_test_features(frame_index=0, face_center=(100.0, 100.0)),
            make_test_features(frame_index=1, face_center=(100.0 + small_movement, 100.0)),
            make_test_features(frame_index=2, face_center=(100.0 + small_movement * 2, 100.0)),
        ]

        unstable_frames = [
            make_test_features(frame_index=0, face_center=(100.0, 100.0)),
            make_test_features(frame_index=1, face_center=(100.0 + large_movement, 100.0)),
            make_test_features(frame_index=2, face_center=(100.0, 100.0)),
        ]

        result_stable = scorer.compute_scores(stable_frames)
        result_unstable = scorer.compute_scores(unstable_frames)

        assert result_stable[1].motion_stability_score == 1.0
        assert result_unstable[1].motion_stability_score == 0.0


class TestSharpnessScoreCriticalPaths:
    def test_all_faces_not_detected_all_sharpness_scores_zero(self, scorer, make_test_features):
        features = [
            make_test_features(frame_index=0, face_detected=False),
            make_test_features(frame_index=1, face_detected=False),
            make_test_features(frame_index=2, face_detected=False),
        ]
        result = scorer.compute_scores(features)

        for score in result:
            assert score.visual_sharpness_score == 0.0

    def test_sharpness_below_threshold_scores_zero(self, scorer, make_test_features):
        features = [
            make_test_features(frame_index=0, sharpness=BELOW_THRESHOLD_SHARPNESS),
            make_test_features(frame_index=1, sharpness=DEFAULT_SHARPNESS),
        ]
        result = scorer.compute_scores(features)

        assert result[0].visual_sharpness_score == 0.0
        assert result[1].visual_sharpness_score > 0.0


class TestFinalScoreCalculation:
    def test_weighted_sum_uses_correct_weights(self, scorer, make_test_features):
        config = RankingConfig()
        features = [
            make_test_features(frame_index=0),
            make_test_features(frame_index=1),
            make_test_features(frame_index=2),
        ]
        result = scorer.compute_scores(features)

        for score in result:
            motion_pose_weight = config.motion_stability_weight + config.pose_stability_weight
            expected = (
                config.eye_openness_weight * score.eye_score
                + config.expression_neutrality_weight * score.mouth_score
                + config.visual_sharpness_weight * score.visual_sharpness_score
                + motion_pose_weight * score.motion_stability_score
            )
            assert abs(score.final_score - expected) < 0.0001
