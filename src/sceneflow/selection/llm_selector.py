import os
import base64
import logging
from typing import List, Optional
import cv2
import numpy as np
from openai import OpenAI

from sceneflow.shared.models import (
    RankedFrame,
    FrameScore,
    FrameFeatures,
    FrameMetadata,
    TemporalContext,
    NormalizedScores,
    RawMeasurements,
)

logger = logging.getLogger(__name__)


class LLMFrameSelector:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        self.client = OpenAI(api_key=self.api_key)

    def select_best_frame(
        self,
        video_path: str,
        ranked_frames: List[RankedFrame],
        speech_end_time: float,
        video_duration: float,
        all_scores: List[FrameScore],
        all_features: List[FrameFeatures]
    ) -> RankedFrame:
        if len(ranked_frames) == 0:
            raise ValueError("No frames provided for selection")

        if len(ranked_frames) == 1:
            return ranked_frames[0]

        frames_data = []
        for idx, frame in enumerate(ranked_frames[:5], 1):
            score = next((s for s in all_scores if s.frame_index == frame.frame_index), None)
            features = next((f for f in all_features if f.frame_index == frame.frame_index), None)

            if not score or not features:
                logger.warning(f"Missing score or features for frame {frame.frame_index}, skipping")
                continue

            image_bytes = self._extract_frame_image(video_path, frame.timestamp)
            metadata = self._build_metadata(frame, score, features, speech_end_time, video_duration)
            frames_data.append({
                "index": idx,
                "frame": frame,
                "image_bytes": image_bytes,
                "metadata": metadata
            })

        if not frames_data:
            raise ValueError("No valid frame data for LLM selection")

        selected_index = self._call_openai_vision(frames_data)
        return frames_data[selected_index]["frame"]

    def _extract_frame_image(self, video_path: str, timestamp: float) -> bytes:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to extract frame at timestamp {timestamp}")

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    def _build_metadata(
        self,
        frame: RankedFrame,
        score: FrameScore,
        features: FrameFeatures,
        speech_end_time: float,
        video_duration: float
    ) -> FrameMetadata:
        time_since_speech = frame.timestamp - speech_end_time
        time_until_end = video_duration - frame.timestamp
        percentage_through = (frame.timestamp / video_duration) * 100

        return FrameMetadata(
            timestamp=round(frame.timestamp, 2),
            overall_score=round(frame.score, 4),
            scores=NormalizedScores(
                eye_openness=round(score.eye_openness_score, 3),
                motion_stability=round(score.motion_stability_score, 3),
                expression_neutrality=round(score.expression_neutrality_score, 3),
                pose_stability=round(score.pose_stability_score, 3),
                visual_sharpness=round(score.visual_sharpness_score, 3)
            ),
            raw_measurements=RawMeasurements(
                eye_aspect_ratio=round(features.eye_openness, 3),
                motion_magnitude=round(features.motion_magnitude, 2),
                mouth_aspect_ratio=round(features.expression_activity, 3),
                head_pose_deviation=round(features.pose_deviation, 2),
                sharpness_variance=round(features.sharpness, 2)
            ),
            temporal_context=TemporalContext(
                time_since_speech_end=round(time_since_speech, 2),
                time_until_video_end=round(time_until_end, 2),
                percentage_through_video=round(percentage_through, 1)
            )
        )

    def _build_prompt(self, frames_data: List[dict]) -> str:
        prompt_parts = [
            "You are analyzing AI-generated talking head video frames to select the best cut point.",
            f"The video is {frames_data[0]['metadata'].temporal_context.time_until_video_end + frames_data[0]['metadata'].timestamp:.1f}s long.",
            f"Speech ended at {frames_data[0]['metadata'].timestamp - frames_data[0]['metadata'].temporal_context.time_since_speech_end:.1f}s.",
            "",
            "Below are 5 frames ranked by an algorithm analyzing facial features and stability.",
            "",
            "For each frame, evaluate:",
            "1. Does the person look natural and alert (not mid-blink, not mid-expression)?",
            "2. Is the mouth in a neutral/closed position (not partially open)?",
            "3. Does the face appear stable (not blurry or in motion)?",
            "4. Is the padding appropriate (not too rushed after speech, not too delayed before video end)?",
            "",
            "Select the single best frame number (1-5) that would create the most natural ending.",
            "Respond with ONLY the number (1-5), nothing else.",
            ""
        ]

        for data in frames_data:
            meta = data["metadata"]
            prompt_parts.extend([
                f"[Frame {data['index']} - {meta.timestamp}s - Score: {meta.overall_score}]",
                f"- Eye openness: {meta.scores.eye_openness} (EAR: {meta.raw_measurements.eye_aspect_ratio})",
                f"- Motion stability: {meta.scores.motion_stability} ({meta.raw_measurements.motion_magnitude}px movement)",
                f"- Expression neutrality: {meta.scores.expression_neutrality} (MAR: {meta.raw_measurements.mouth_aspect_ratio})",
                f"- Pose stability: {meta.scores.pose_stability} ({meta.raw_measurements.head_pose_deviation}px deviation)",
                f"- Visual sharpness: {meta.scores.visual_sharpness}",
                f"- Padding: {meta.temporal_context.time_since_speech_end}s after speech, {meta.temporal_context.time_until_video_end}s before end ({meta.temporal_context.percentage_through_video}% through video)",
                ""
            ])

        return "\n".join(prompt_parts)

    def _call_openai_vision(self, frames_data: List[dict]) -> int:
        prompt = self._build_prompt(frames_data)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        for data in frames_data:
            base64_image = base64.b64encode(data["image_bytes"]).decode('utf-8')
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            })

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0
        )

        response_text = response.choices[0].message.content.strip()

        try:
            selected_number = int(response_text)
            if 1 <= selected_number <= len(frames_data):
                logger.debug(f"LLM selected frame {selected_number}")
                return selected_number - 1
            else:
                logger.warning(f"LLM returned invalid frame number: {selected_number}, using top algorithmic result")
                return 0
        except ValueError:
            logger.warning(f"LLM returned non-numeric response: {response_text}, using top algorithmic result")
            return 0
