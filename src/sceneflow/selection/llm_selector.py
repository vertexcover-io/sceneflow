import os
import base64
import logging
from typing import List, Optional

from openai import OpenAI

from sceneflow.shared.models import RankedFrame
from sceneflow.utils.video import extract_frame_at_timestamp

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
    ) -> RankedFrame:
        if len(ranked_frames) == 0:
            raise ValueError("No frames provided for selection")

        if len(ranked_frames) == 1:
            return ranked_frames[0]

        frames_data = []
        for idx, frame in enumerate(ranked_frames[:5], 1):
            image_bytes = self._extract_frame_image(video_path, frame.timestamp)
            metadata = self._build_metadata(frame, speech_end_time, video_duration)
            frames_data.append({
                "index": idx,
                "frame": frame,
                "image_bytes": image_bytes,
                "metadata": metadata
            })

        if not frames_data:
            raise ValueError("No valid frame data for LLM selection")

        selected_index = self._call_openai_vision(frames_data, video_duration, speech_end_time)
        return frames_data[selected_index]["frame"]

    def _extract_frame_image(self, video_path: str, timestamp: float) -> bytes:
        return extract_frame_at_timestamp(video_path, timestamp)

    def _build_metadata(
        self,
        frame: RankedFrame,
        speech_end_time: float,
        video_duration: float
    ) -> dict:
        return {
            "timestamp": round(frame.timestamp, 2),
            "score": round(frame.score, 2),
            "time_since_speech": round(frame.timestamp - speech_end_time, 2),
            "time_until_end": round(video_duration - frame.timestamp, 2),
        }

    def _build_prompt(self, frames_data: List[dict], video_duration: float, speech_end_time: float) -> str:
        prompt_parts = [
            "Select the best cut point from these frames of a talking head video.",
            f"Video: {video_duration:.1f}s, Speech ended: {speech_end_time:.1f}s",
            "",
            "Evaluate each frame visually for:",
            "- Natural, alert expression (not mid-blink)",
            "- Mouth closed/neutral (not speaking)",
            "- Clear, not blurry",
            "- Good padding after speech end",
            "",
            "Frames:",
        ]

        for data in frames_data:
            meta = data["metadata"]
            prompt_parts.append(
                f"[Frame {data['index']}] {meta['timestamp']}s | Score: {meta['score']} | +{meta['time_since_speech']}s after speech, -{meta['time_until_end']}s before end"
            )

        prompt_parts.append("")
        prompt_parts.append("Reply with ONLY the frame number (1-5).")

        return "\n".join(prompt_parts)

    def _call_openai_vision(self, frames_data: List[dict], video_duration: float, speech_end_time: float) -> int:
        prompt = self._build_prompt(frames_data, video_duration, speech_end_time)

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
                    "detail": "low"
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
