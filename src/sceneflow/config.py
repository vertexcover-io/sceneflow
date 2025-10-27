from dataclasses import dataclass


@dataclass
class RankingConfig:
    eye_openness_weight: float = 0.30
    motion_stability_weight: float = 0.25
    expression_neutrality_weight: float = 0.20
    pose_stability_weight: float = 0.15
    visual_sharpness_weight: float = 0.10

    context_window_size: int = 5

    # Multi-stage ranking parameters
    quality_gate_percentile: float = 75.0
    local_stability_window: int = 5

    # Multi-face detection parameters
    center_weighting_strength: float = 1.0      # Controls center distance weighting (higher = stronger center bias)
    min_face_confidence: float = 0.5            # Minimum confidence for face detection (0-1)

    def validate(self):
        total = (
            self.eye_openness_weight +
            self.motion_stability_weight +
            self.expression_neutrality_weight +
            self.pose_stability_weight +
            self.visual_sharpness_weight
        )
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"
        assert self.context_window_size % 2 == 1, "Context window size must be odd"
        assert self.local_stability_window % 2 == 1, "Stability window size must be odd"
