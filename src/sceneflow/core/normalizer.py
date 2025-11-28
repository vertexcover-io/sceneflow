import numpy as np
from typing import List


class MetricNormalizer:
    """
    Advanced normalization methods for converting raw feature values to scores.

    Provides multiple normalization strategies optimized for different feature types.
    All normalized scores are in [0, 1] range where higher = better for cut points.
    """

    @staticmethod
    def normalize(values: List[float], inverse: bool = False) -> List[float]:
        """
        Min-max normalization to [0, 1] range.

        Args:
            values: Raw metric values
            inverse: If True, invert the normalization (higher raw = lower score)
                    Use inverse=True for penalty metrics (motion, expression activity)
                    Use inverse=False for reward metrics (sharpness)

        Returns:
            List of normalized scores in [0, 1] range
        """
        if not values or len(values) == 1:
            return [0.5] * len(values)

        arr = np.array(values)
        min_val = arr.min()
        max_val = arr.max()

        # No variation - all values are the same
        if max_val - min_val < 1e-9:
            return [0.5] * len(values)

        # Standard min-max normalization
        normalized = (arr - min_val) / (max_val - min_val)

        # Invert if this is a penalty metric
        if inverse:
            normalized = 1.0 - normalized

        return normalized.tolist()

    @staticmethod
    def normalize_with_deviation(values: List[float], target_value: float,
                                 inverse: bool = False) -> List[float]:
        """
        Normalize based on deviation from a target value.

        Scores frames by how close they are to a target (e.g., median).
        Lower deviation = higher score (better).

        This is ideal for features where we want "normal" values, not extremes:
        - Eye openness (prefer normal, not too wide or too closed)
        - Pose stability (prefer centered, not extreme angles)

        Args:
            values: Raw metric values
            target_value: The ideal/target value (usually median)
            inverse: If True, higher deviation = higher score (rarely used)

        Returns:
            List of normalized scores in [0, 1] range
            Higher score = closer to target (better)
        """
        if not values:
            return []

        # Calculate absolute deviation from target
        deviations = [abs(v - target_value) for v in values]

        # Normalize deviations (high deviation = high normalized value)
        normalized_deviations = MetricNormalizer.normalize(deviations, inverse=False)

        # Invert: we want LOW deviation to get HIGH score
        if not inverse:
            scores = [1.0 - d for d in normalized_deviations]
        else:
            scores = normalized_deviations

        return scores

    @staticmethod
    def normalize_robust(values: List[float], inverse: bool = False,
                        percentile_low: float = 5.0,
                        percentile_high: float = 95.0) -> List[float]:
        """
        Robust normalization using percentiles instead of min/max.

        Reduces impact of outliers by clipping to percentile range before normalizing.

        Args:
            values: Raw metric values
            inverse: If True, invert the normalization
            percentile_low: Lower percentile for clipping (default: 5th)
            percentile_high: Upper percentile for clipping (default: 95th)

        Returns:
            List of normalized scores in [0, 1] range
        """
        if not values or len(values) == 1:
            return [0.5] * len(values)

        arr = np.array(values)

        # Calculate percentile bounds
        p_low = np.percentile(arr, percentile_low)
        p_high = np.percentile(arr, percentile_high)

        # Clip to percentile range
        clipped = np.clip(arr, p_low, p_high)

        # Normalize
        if p_high - p_low < 1e-9:
            return [0.5] * len(values)

        normalized = (clipped - p_low) / (p_high - p_low)

        if inverse:
            normalized = 1.0 - normalized

        return normalized.tolist()

    @staticmethod
    def normalize_sigmoid(values: List[float], midpoint: float = None,
                         steepness: float = 10.0, inverse: bool = False) -> List[float]:
        """
        Sigmoid (S-curve) normalization for smooth transitions.

        Provides gradual scoring near the midpoint and sharp penalties at extremes.

        Args:
            values: Raw metric values
            midpoint: Center point of sigmoid. If None, uses median.
            steepness: How steep the S-curve is (higher = more binary)
            inverse: If True, invert the curve

        Returns:
            List of normalized scores in [0, 1] range
        """
        if not values:
            return []

        arr = np.array(values)

        if midpoint is None:
            midpoint = float(np.median(arr))

        # Scale values to roughly [-3, 3] range for sigmoid
        arr_range = arr.max() - arr.min()
        if arr_range < 1e-9:
            return [0.5] * len(values)

        scaled = ((arr - midpoint) / arr_range) * steepness

        # Apply sigmoid function
        sigmoid = 1.0 / (1.0 + np.exp(-scaled))

        if inverse:
            sigmoid = 1.0 - sigmoid

        return sigmoid.tolist()

    @staticmethod
    def normalize_gaussian(values: List[float], target: float = None,
                          sigma: float = None, inverse: bool = False) -> List[float]:
        """
        Gaussian (bell curve) normalization.

        Rewards values near the target, penalizes values far from target.
        Creates smooth scoring with maximum score at the target value.

        Ideal for features where we want a specific value (e.g., normal eye openness).

        Args:
            values: Raw metric values
            target: Peak of the curve. If None, uses median.
            sigma: Standard deviation (width of curve). If None, uses data std.
            inverse: If True, invert (penalize values near target)

        Returns:
            List of normalized scores in [0, 1] range
            Higher score = closer to target
        """
        if not values:
            return []

        arr = np.array(values)

        if target is None:
            target = float(np.median(arr))

        if sigma is None:
            sigma = float(np.std(arr))
            if sigma < 1e-9:
                sigma = 1.0

        # Gaussian formula: exp(-((x - target)^2) / (2 * sigma^2))
        squared_deviations = (arr - target) ** 2
        gaussian = np.exp(-squared_deviations / (2 * sigma ** 2))

        if inverse:
            gaussian = 1.0 - gaussian

        return gaussian.tolist()

    @staticmethod
    def normalize_percentile_rank(values: List[float], inverse: bool = False) -> List[float]:
        """
        Rank-based normalization using percentile ranks.

        Converts values to their percentile position (0-100, scaled to 0-1).
        More robust to outliers than min-max normalization.

        Args:
            values: Raw metric values
            inverse: If True, invert the ranks

        Returns:
            List of normalized scores in [0, 1] range
        """
        if not values:
            return []

        if len(values) == 1:
            return [0.5]

        arr = np.array(values)

        # Calculate percentile rank for each value using numpy only
        # For each value, count how many values are <= it
        percentiles = np.zeros(len(arr))
        for i, val in enumerate(arr):
            percentiles[i] = (arr <= val).sum() / len(arr) * 100.0

        # Convert from [0, 100] to [0, 1]
        normalized = percentiles / 100.0

        if inverse:
            normalized = 1.0 - normalized

        return normalized.tolist()

    @staticmethod
    def normalize_zscore_clipped(values: List[float], inverse: bool = False,
                                clip_std: float = 3.0) -> List[float]:
        """
        Z-score normalization with clipping.

        Standardizes values to mean=0, std=1, then clips to ±clip_std and scales to [0,1].
        Good for handling outliers while preserving relative differences.

        Args:
            values: Raw metric values
            inverse: If True, invert the normalization
            clip_std: Number of standard deviations to clip at

        Returns:
            List of normalized scores in [0, 1] range
        """
        if not values or len(values) == 1:
            return [0.5] * len(values)

        arr = np.array(values)
        mean = arr.mean()
        std = arr.std()

        if std < 1e-9:
            return [0.5] * len(values)

        # Standardize to z-scores
        z_scores = (arr - mean) / std

        # Clip to ±clip_std
        clipped = np.clip(z_scores, -clip_std, clip_std)

        # Scale to [0, 1]
        normalized = (clipped + clip_std) / (2 * clip_std)

        if inverse:
            normalized = 1.0 - normalized

        return normalized.tolist()
