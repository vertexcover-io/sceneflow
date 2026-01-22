from dataclasses import dataclass


@dataclass
class RankingConfig:
    """Config for podcast/talking head cut point detection."""

    eye_openness_weight: float = 0.30
    motion_stability_weight: float = 0.25
    expression_neutrality_weight: float = 0.20
    pose_stability_weight: float = 0.15
    visual_sharpness_weight: float = 0.10

    context_window_size: int = 7
    quality_gate_percentile: float = 80.0
    local_stability_window: int = 7

    min_sharpness: float = 50.0
    max_position_delta: float = 30.0

    min_face_confidence: float = 0.5

    def validate(self):
        total = (
            self.eye_openness_weight
            + self.motion_stability_weight
            + self.expression_neutrality_weight
            + self.pose_stability_weight
            + self.visual_sharpness_weight
        )
        if abs(total - 1.0) >= 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        if self.context_window_size % 2 == 0:
            raise ValueError("context_window_size must be odd")

        if self.local_stability_window % 2 == 0:
            raise ValueError("local_stability_window must be odd")
