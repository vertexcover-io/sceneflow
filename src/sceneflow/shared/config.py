from dataclasses import dataclass


@dataclass
class RankingConfig:
    """Config for podcast/talking head cut point detection."""
    
    # Weights for scoring (must sum to 1.0)
    eye_weight: float = 0.30       # Penalize blinks
    mouth_weight: float = 0.40     # Penalize open mouth - most important
    sharpness_weight: float = 0.15  # Penalize blurry/motion blur frames
    consistency_weight: float = 0.15  # Penalize sudden movements
    
    # Thresholds
    min_sharpness: float = 50.0    # Minimum acceptable sharpness
    max_position_delta: float = 30.0  # Max face movement in pixels for consistency
    
    # Face detection
    min_face_confidence: float = 0.5

    def validate(self):
        total = self.eye_weight + self.mouth_weight + self.sharpness_weight + self.consistency_weight
        if abs(total - 1.0) >= 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
