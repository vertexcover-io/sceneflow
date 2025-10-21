from typing import Optional
import whisper


class SpeechTimestampDetector:
    """Detects when speech ends in video/audio files using Whisper."""

    def __init__(self, model_size: str = "base"):
        """
        Initialize the detector with a Whisper model.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
                       Larger models are more accurate but slower.
        """
        self.model = whisper.load_model(model_size)

    def get_speech_end_time(self, file_path: str) -> Optional[float]:
        """
        Get the timestamp when speech ends in the file.

        Args:
            file_path: Path to video or audio file

        Returns:
            End timestamp in seconds, or None if no speech detected
        """
        result = self.model.transcribe(file_path, word_timestamps=True)

        if not result['segments']:
            return None

        return result['segments'][-1]['end']

    def get_all_speech_segments(self, file_path: str) -> list[dict]:
        """
        Get all speech segments with start and end timestamps.

        Args:
            file_path: Path to video or audio file

        Returns:
            List of segments with 'start', 'end', and 'text' keys
        """
        result = self.model.transcribe(file_path, word_timestamps=True)

        return [
            {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            }
            for segment in result['segments']
        ]
