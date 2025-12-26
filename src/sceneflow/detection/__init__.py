"""Speech and audio detection package.

This package provides modules for detecting speech end times and refining
them using audio energy analysis.
"""

from sceneflow.detection.speech_detector import SpeechDetector
from sceneflow.detection.energy_refiner import refine_speech_end

__all__ = [
    "SpeechDetector",
    "refine_speech_end",
]
