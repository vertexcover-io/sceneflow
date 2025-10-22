from typing import Optional, Literal
import whisper
import json
from pathlib import Path
import numpy as np
import librosa
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


class SpeechTimestampDetector:
    """Detects when speech ends in video/audio files using Whisper and optional VAD."""

    def __init__(self, model_size: str = "small", smoothing_window: int = 5, min_drop_ratio: float = 0.1, use_vad: bool = True):
        """
        Initialize the detector with a Whisper model and optional Silero VAD.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
                       Larger models are more accurate but slower.
            smoothing_window: Number of frames to smooth energy curve (default: 5)
                            Larger = smoother but less precise, Smaller = more sensitive to noise
            min_drop_ratio: Minimum energy drop as ratio of peak energy (default: 0.1 = 10%)
                          Set to 0.0 to detect any drop, higher values require larger drops
            use_vad: If True, load Silero VAD model for advanced detection modes (default: True)
                    Set to False to save memory if only using basic Whisper detection
        """
        self.model = whisper.load_model(model_size)
        self.smoothing_window = smoothing_window
        self.min_drop_ratio = min_drop_ratio

        # Load Silero VAD model if requested
        self.vad_model = None
        if use_vad:
            try:
                self.vad_model = load_silero_vad()
            except Exception as e:
                print(f"Warning: Failed to load Silero VAD model ({str(e)}). VAD-based methods will not be available.")

    def _load_audio_for_vad(self, file_path: str) -> torch.Tensor:
        """
        Load audio file for VAD processing using librosa (avoids torchaudio backend issues).

        Args:
            file_path: Path to audio/video file

        Returns:
            Audio as PyTorch tensor at 16kHz sample rate
        """
        # Load audio using librosa (handles both audio and video files)
        # Silero VAD expects 16kHz mono audio
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        # Convert numpy array to PyTorch tensor
        audio_tensor = torch.from_numpy(audio).float()

        return audio_tensor

    def get_speech_end_time(
        self,
        file_path: str,
        return_full_result: bool = False,
        save_transcription: bool = False,
        output_dir: Optional[str] = None
    ) -> Optional[float] | tuple[Optional[float], dict]:
        """
        Get the timestamp when speech ends in the file.

        Args:
            file_path: Path to video or audio file
            return_full_result: If True, also return the full transcription result
            save_transcription: If True, save transcription data to JSON file
            output_dir: Directory to save transcription (default: same as video file)

        Returns:
            End timestamp in seconds, or None if no speech detected
            If return_full_result=True, returns tuple of (end_time, full_result)
        """
        result = self.model.transcribe(file_path, word_timestamps=True)

        # Save transcription if requested
        if save_transcription:
            self._save_transcription(file_path, result, output_dir)

        if not result['segments']:
            if return_full_result:
                return None, result
            return None

        end_time = result['segments'][-1]['end']

        if return_full_result:
            return end_time, result

        return end_time

    def _save_transcription(self, file_path: str, result: dict, output_dir: Optional[str] = None) -> None:
        """
        Save transcription data to a JSON file.

        Args:
            file_path: Path to input video/audio file
            result: Whisper transcription result
            output_dir: Directory to save transcription (default: same as video file)
        """
        video_path = Path(file_path)
        video_base_name = video_path.stem

        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = video_path.parent

        out_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{video_base_name}_transcription.json"
        output_path = out_dir / output_filename

        # Prepare transcription data
        transcription_data = {
            'text': result.get('text', ''),
            'language': result.get('language', 'unknown'),
            'segments': [
                {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'words': segment.get('words', [])
                }
                for segment in result.get('segments', [])
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        print(f"Saved transcription to: {output_path}")

    def _calculate_spectral_confidence(self, audio: np.ndarray, sr: int, timestamp: float, window_size: float = 0.5) -> float:
        """
        Calculate confidence score using spectral features (ZCR and spectral centroid).
        Higher scores indicate more likely speech, lower scores indicate silence.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            timestamp: The timestamp to analyze around
            window_size: Window size in seconds to analyze (default: 0.5s)

        Returns:
            Confidence score between 0 (silence) and 1 (speech)
        """
        # Extract audio window around timestamp
        center_sample = int(timestamp * sr)
        window_samples = int(window_size * sr)
        start = max(0, center_sample - window_samples // 2)
        end = min(len(audio), center_sample + window_samples // 2)

        audio_window = audio[start:end]

        if len(audio_window) < 100:  # Too short to analyze
            return 0.5

        # Calculate Zero Crossing Rate (ZCR)
        # Lower ZCR typically indicates silence or voiced speech
        zcr = librosa.feature.zero_crossing_rate(audio_window)[0]
        zcr_mean = np.mean(zcr)

        # Calculate Spectral Centroid
        # Lower values indicate silence or low-frequency content
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_window, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroid)

        # Normalize features to 0-1 range (approximate ranges)
        # Typical ZCR for speech: 0.05-0.15, silence: < 0.05
        zcr_normalized = min(1.0, max(0.0, (zcr_mean - 0.02) / 0.15))

        # Typical spectral centroid for speech: 1000-3000 Hz, silence: < 500 Hz
        spectral_normalized = min(1.0, max(0.0, (spectral_centroid_mean - 200) / 2800))

        # Combine features (weighted average)
        confidence = 0.6 * (1 - zcr_normalized) + 0.4 * spectral_normalized

        return float(confidence)

    def _find_silence_point(self, audio: np.ndarray, sr: int, start_time: float, search_window: float = 2.0) -> tuple[float, float]:
        """
        Find the exact point where speech ends by detecting energy drop.
        Searches FORWARD from the Whisper timestamp to find where energy decreases.
        NO THRESHOLDS - finds the nearest point with significant energy drop.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            start_time: Starting point (from Whisper) to search forward from
            search_window: How many seconds AFTER start_time to search

        Returns:
            Tuple of (silence_start_time, confidence_score)
        """
        # Define search range - only search forward from start_time
        start_sample = int(start_time * sr)
        end_sample = min(len(audio), int((start_time + search_window) * sr))

        # Extract audio segment to analyze
        audio_segment = audio[start_sample:end_sample]

        # Calculate frame-level energy (RMS) with small hop size for precision
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop for fine-grained detection

        rms = librosa.feature.rms(y=audio_segment, frame_length=frame_length, hop_length=hop_length)[0]

        # Handle edge case: no energy
        if len(rms) < 2 or rms.max() == 0:
            return start_time, 0.0

        # Apply smoothing to reduce noise (moving average)
        if self.smoothing_window > 1 and len(rms) > self.smoothing_window:
            rms_smoothed = np.convolve(rms, np.ones(self.smoothing_window) / self.smoothing_window, mode='same')
        else:
            rms_smoothed = rms

        # Calculate gradient (rate of change in energy)
        # Negative gradient = energy is decreasing
        gradient = np.gradient(rms_smoothed)

        # Find peak energy in the search window to calculate minimum drop threshold
        peak_energy = rms_smoothed.max()
        min_drop_magnitude = peak_energy * self.min_drop_ratio

        # Find the first significant energy drop
        # Look for negative gradient with sufficient magnitude
        drop_frame = None
        for i in range(len(gradient)):
            # Check if energy is dropping (negative gradient)
            # and the drop magnitude is significant enough
            if gradient[i] < 0 and abs(gradient[i]) > min_drop_magnitude:
                drop_frame = i
                break

        # If no significant drop found, look for the steepest drop overall
        if drop_frame is None:
            # Find the most negative gradient (steepest drop)
            min_gradient_idx = np.argmin(gradient)
            if gradient[min_gradient_idx] < 0:
                drop_frame = min_gradient_idx

        # Convert frame index to timestamp
        if drop_frame is not None:
            silence_time_offset = (drop_frame * hop_length) / sr
            silence_time = (start_sample / sr) + silence_time_offset

            # Calculate confidence based on the magnitude of the drop
            # Compare energy before and after the drop point
            window = min(5, drop_frame)  # Look at 5 frames before/after
            if drop_frame >= window and drop_frame + window < len(rms_smoothed):
                energy_before = np.mean(rms_smoothed[drop_frame - window:drop_frame])
                energy_after = np.mean(rms_smoothed[drop_frame:drop_frame + window])

                # Confidence is the relative drop in energy
                if energy_before > 0:
                    confidence = min(1.0, (energy_before - energy_after) / energy_before)
                else:
                    confidence = 0.5
            else:
                confidence = 0.5

            return silence_time, confidence

        # No drop found at all, return start time with low confidence
        return start_time, 0.0

    def get_precise_speech_end_time(
        self,
        file_path: str,
        search_window: float = 2.0,
        return_confidence: bool = False,
        save_transcription: bool = False,
        output_dir: Optional[str] = None
    ) -> float | tuple[float, float]:
        """
        Get the precise timestamp when speech ends using hybrid approach:
        1. Use Whisper to get approximate end time from transcription
        2. Analyze audio energy FORWARD from that point to find exact silence

        Args:
            file_path: Path to video or audio file
            search_window: Seconds to search FORWARD from Whisper timestamp (default: 2.0)
            return_confidence: If True, return (timestamp, confidence) tuple
            save_transcription: If True, save transcription data to JSON file
            output_dir: Directory to save transcription (default: same as video file)

        Returns:
            Precise end timestamp in seconds
            If return_confidence=True, returns (timestamp, confidence_score)
        """
        # Step 1: Get Whisper's transcription with word timestamps
        result = self.model.transcribe(file_path, word_timestamps=True)

        # Save transcription if requested
        if save_transcription:
            self._save_transcription(file_path, result, output_dir)

        if not result['segments']:
            if return_confidence:
                return None, 0.0
            return None

        # Get the last word's end timestamp from Whisper
        last_segment = result['segments'][-1]
        if 'words' in last_segment and last_segment['words']:
            # Use last word timestamp for more precision
            whisper_end_time = last_segment['words'][-1]['end']
        else:
            # Fall back to segment end time
            whisper_end_time = last_segment['end']

        # Step 2: Load audio for precise analysis
        try:
            # Load audio (librosa automatically handles video files by extracting audio)
            audio, sr = librosa.load(file_path, sr=16000, mono=True)

            # Find exact silence point around Whisper's estimate
            precise_end_time, confidence = self._find_silence_point(
                audio, sr, whisper_end_time, search_window
            )

            if return_confidence:
                return precise_end_time, confidence
            return precise_end_time

        except Exception as e:
            print(f"Warning: Audio analysis failed ({str(e)}), falling back to Whisper timestamp")
            if return_confidence:
                return whisper_end_time, 0.5  # Medium confidence for fallback
            return whisper_end_time

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

    def get_full_transcription(self, file_path: str) -> dict:
        """
        Get the full transcription result from Whisper.

        Args:
            file_path: Path to video or audio file

        Returns:
            Complete transcription result dictionary containing:
            - 'text': Full transcription text
            - 'segments': List of segments with timestamps and text
            - 'language': Detected language
        """
        result = self.model.transcribe(file_path, word_timestamps=True)

        return {
            'text': result['text'],
            'language': result['language'],
            'segments': [
                {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'words': segment.get('words', [])
                }
                for segment in result['segments']
            ]
        }

    def get_vad_speech_end_time(
        self,
        file_path: str,
        return_confidence: bool = False,
        save_transcription: bool = False,
        output_dir: Optional[str] = None
    ) -> float | tuple[float, float]:
        """
        Get speech end time using Silero VAD (Voice Activity Detection).
        This method uses deep learning for highly accurate speech/silence detection.

        Args:
            file_path: Path to video or audio file
            return_confidence: If True, return (timestamp, confidence) tuple
            save_transcription: If True, save transcription data to JSON file
            output_dir: Directory to save transcription (default: same as video file)

        Returns:
            Speech end timestamp in seconds
            If return_confidence=True, returns (timestamp, confidence_score)

        Raises:
            RuntimeError: If VAD model is not loaded (use_vad=False in __init__)
        """
        if self.vad_model is None:
            raise RuntimeError("VAD model not loaded. Initialize detector with use_vad=True")

        # Get transcription if needed
        if save_transcription:
            result = self.model.transcribe(file_path, word_timestamps=True)
            self._save_transcription(file_path, result, output_dir)

        try:
            # Load audio using our librosa-based helper (avoids torchaudio backend issues)
            wav = self._load_audio_for_vad(file_path)

            # Get speech timestamps using VAD
            speech_timestamps = get_speech_timestamps(
                wav,
                self.vad_model,
                return_seconds=True
            )

            if not speech_timestamps:
                # No speech detected
                if return_confidence:
                    return 0.0, 0.0
                return 0.0

            # Get the end time of the last speech segment
            last_speech_segment = speech_timestamps[-1]
            vad_end_time = last_speech_segment['end']

            # Calculate confidence based on VAD's detection quality
            # Higher confidence if the last segment is well-defined
            segment_duration = last_speech_segment['end'] - last_speech_segment['start']
            confidence = min(1.0, segment_duration / 0.5)  # 0.5s or longer = full confidence

            if return_confidence:
                return vad_end_time, confidence
            return vad_end_time

        except Exception as e:
            print(f"Warning: VAD analysis failed ({str(e)}), falling back to Whisper")
            # Fallback to basic Whisper detection
            result = self.model.transcribe(file_path, word_timestamps=True)
            if not result['segments']:
                if return_confidence:
                    return None, 0.0
                return None

            whisper_end = result['segments'][-1]['end']
            if return_confidence:
                return whisper_end, 0.5
            return whisper_end

    def get_hybrid_speech_end_time(
        self,
        file_path: str,
        search_window: float = 1.0,
        return_confidence: bool = False,
        return_details: bool = False,
        save_transcription: bool = False,
        output_dir: Optional[str] = None
    ) -> float | tuple[float, float] | tuple[float, dict]:
        """
        Get speech end time using hybrid approach combining Whisper, Silero VAD, and energy analysis.
        This is the most accurate method, combining multiple detection strategies.

        Strategy:
        1. Whisper: Get approximate end time from transcription
        2. Silero VAD: Find precise speech boundaries in nearby region
        3. Energy analysis: Refine to exact silence point
        4. Spectral features: Validate with ZCR and spectral centroid

        Args:
            file_path: Path to video or audio file
            search_window: Seconds to search around Whisper timestamp (default: 1.0)
            return_confidence: If True, return (timestamp, confidence) tuple
            return_details: If True, return (timestamp, details_dict) with all method results
            save_transcription: If True, save transcription data to JSON file
            output_dir: Directory to save transcription (default: same as video file)

        Returns:
            Precise speech end timestamp in seconds
            If return_confidence=True, returns (timestamp, confidence_score)
            If return_details=True, returns (timestamp, details_dict)

        Raises:
            RuntimeError: If VAD model is not loaded (use_vad=False in __init__)
        """
        if self.vad_model is None:
            raise RuntimeError("VAD model not loaded. Initialize detector with use_vad=True")

        # Step 1: Get Whisper transcription
        result = self.model.transcribe(file_path, word_timestamps=True)

        if save_transcription:
            self._save_transcription(file_path, result, output_dir)

        if not result['segments']:
            if return_details:
                return None, {'error': 'No speech detected by Whisper'}
            if return_confidence:
                return None, 0.0
            return None

        # Get Whisper's estimate
        last_segment = result['segments'][-1]
        if 'words' in last_segment and last_segment['words']:
            whisper_end_time = last_segment['words'][-1]['end']
        else:
            whisper_end_time = last_segment['end']

        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000, mono=True)

            # Step 2: Get VAD timestamps around Whisper estimate
            # Use our librosa-based helper (avoids torchaudio backend issues)
            wav = self._load_audio_for_vad(file_path)
            speech_timestamps = get_speech_timestamps(
                wav,
                self.vad_model,
                return_seconds=True
            )

            # Find the VAD segment closest to Whisper's estimate
            vad_end_time = whisper_end_time  # Default to Whisper
            if speech_timestamps:
                # Find last speech segment that ends near Whisper's estimate
                for segment in reversed(speech_timestamps):
                    if abs(segment['end'] - whisper_end_time) <= search_window:
                        vad_end_time = segment['end']
                        break
                else:
                    # If no close match, use the last segment
                    vad_end_time = speech_timestamps[-1]['end']

            # Step 3: Refine with energy gradient analysis
            energy_end_time, energy_confidence = self._find_silence_point(
                audio, sr, vad_end_time, search_window=search_window
            )

            # Step 4: Validate with spectral features
            spectral_confidence = self._calculate_spectral_confidence(
                audio, sr, energy_end_time, window_size=0.3
            )

            # Combine confidence scores
            # High confidence if all methods agree
            time_agreement = abs(whisper_end_time - vad_end_time) + abs(vad_end_time - energy_end_time)
            agreement_score = max(0.0, 1.0 - (time_agreement / search_window))

            final_confidence = (
                0.3 * agreement_score +
                0.3 * energy_confidence +
                0.2 * spectral_confidence +
                0.2  # Base confidence for using multiple methods
            )
            final_confidence = min(1.0, final_confidence)

            # Use energy-refined timestamp as final result
            final_timestamp = energy_end_time

            if return_details:
                details = {
                    'whisper_end': whisper_end_time,
                    'vad_end': vad_end_time,
                    'energy_end': energy_end_time,
                    'final_end': final_timestamp,
                    'energy_confidence': energy_confidence,
                    'spectral_confidence': spectral_confidence,
                    'agreement_score': agreement_score,
                    'final_confidence': final_confidence
                }
                return final_timestamp, details

            if return_confidence:
                return final_timestamp, final_confidence

            return final_timestamp

        except Exception as e:
            print(f"Warning: Hybrid analysis failed ({str(e)}), falling back to precise mode")
            # Fallback to precise energy-based method
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            precise_end, confidence = self._find_silence_point(
                audio, sr, whisper_end_time, search_window
            )

            if return_details:
                return precise_end, {'error': str(e), 'fallback': 'precise', 'confidence': confidence}
            if return_confidence:
                return precise_end, confidence
            return precise_end
