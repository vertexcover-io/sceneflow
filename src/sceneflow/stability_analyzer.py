import numpy as np
from typing import List
from .models import FrameFeatures


class StabilityAnalyzer:
    def __init__(self, window_size: int = 5):
        assert window_size % 2 == 1, "Window size must be odd"
        self.window_size = window_size

    def calculate_stability_boosts(self, features: List[FrameFeatures]) -> List[float]:
        if not features:
            return []

        half_window = self.window_size // 2
        motion_values = [f.motion_magnitude for f in features]
        pose_values = [f.pose_deviation for f in features]

        motion_variances = []
        pose_variances = []

        for i in range(len(features)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(features), i + half_window + 1)

            motion_window = motion_values[start_idx:end_idx]
            pose_window = pose_values[start_idx:end_idx]

            motion_var = np.var(motion_window) if len(motion_window) > 1 else 0.0
            pose_var = np.var(pose_window) if len(pose_window) > 1 else 0.0

            motion_variances.append(motion_var)
            pose_variances.append(pose_var)

        motion_var_array = np.array(motion_variances)
        pose_var_array = np.array(pose_variances)

        motion_normalized = self._normalize_variance(motion_var_array)
        pose_normalized = self._normalize_variance(pose_var_array)

        boosts = []
        for motion_norm, pose_norm in zip(motion_normalized, pose_normalized):
            combined_stability = (motion_norm + pose_norm) / 2.0
            boost = 1.0 + (0.5 * combined_stability)
            boosts.append(boost)

        return boosts

    def _normalize_variance(self, variances: np.ndarray) -> np.ndarray:
        if len(variances) == 0:
            return variances

        max_var = np.percentile(variances, 95)
        if max_var < 1e-9:
            return np.ones_like(variances)

        normalized = 1.0 - np.clip(variances / max_var, 0.0, 1.0)
        return normalized
