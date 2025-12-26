import asyncio
import os
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from openai import OpenAI, AsyncOpenAI

from sceneflow.shared.models import RankedFrame
from sceneflow.utils.video import extract_frame_at_timestamp

logger = logging.getLogger(__name__)


class LLMFrameSelector:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        self.client = OpenAI(api_key=self.api_key)
        self._async_client: Optional[AsyncOpenAI] = None

    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    def select_best_frame(
        self,
        video_path: str,
        ranked_frames: List[RankedFrame],
        speech_end_time: float,
        video_duration: float,
    ) -> RankedFrame:
        """Select the best frame from top candidates using LLM vision analysis.

        Uses parallel frame extraction to speed up image loading. Each extraction
        opens its own VideoCapture instance, which is thread-safe since they don't
        share state.
        """
        if len(ranked_frames) == 0:
            raise ValueError("No frames provided for selection")

        if len(ranked_frames) == 1:
            return ranked_frames[0]

        # Take top 5 candidates for LLM analysis
        candidates = ranked_frames[:5]
        frames_data = self._extract_frames_parallel(
            video_path, candidates, speech_end_time, video_duration
        )

        if not frames_data:
            raise ValueError("No valid frame data for LLM selection")

        selected_index = self._call_openai_vision(frames_data, video_duration, speech_end_time)
        return frames_data[selected_index]["frame"]

    def _extract_frames_parallel(
        self,
        video_path: str,
        candidates: List[RankedFrame],
        speech_end_time: float,
        video_duration: float,
    ) -> List[dict]:
        """Extract frame images in parallel using ThreadPoolExecutor.

        Returns a list of frame data dicts with index, frame, image_bytes, and metadata,
        ordered by the original candidate ranking (preserves order).
        """
        num_frames = len(candidates)
        max_workers = min(num_frames, os.cpu_count() or 4)

        def extract_single(args: Tuple[int, RankedFrame]) -> Tuple[int, dict]:
            idx, frame = args
            try:
                image_bytes = self._extract_frame_image(video_path, frame.timestamp)
                metadata = self._build_metadata(frame, speech_end_time, video_duration)
                return (
                    idx,
                    {
                        "index": idx,
                        "frame": frame,
                        "image_bytes": image_bytes,
                        "metadata": metadata,
                    },
                )
            except Exception as e:
                logger.error("Failed to extract frame at %.4fs: %s", frame.timestamp, e)
                return (idx, None)

        # Submit all extraction tasks
        results: List[Tuple[int, dict]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(idx, frame) for idx, frame in enumerate(candidates, 1)]
            futures = [executor.submit(extract_single, task) for task in tasks]

            for future in as_completed(futures):
                result = future.result()
                if result[1] is not None:
                    results.append(result)

        # Sort by original index to preserve ranking order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _extract_frame_image(self, video_path: str, timestamp: float) -> bytes:
        return extract_frame_at_timestamp(video_path, timestamp)

    def _build_metadata(
        self, frame: RankedFrame, speech_end_time: float, video_duration: float
    ) -> dict:
        return {
            "timestamp": round(frame.timestamp, 2),
            "score": round(frame.score, 2),
            "time_since_speech": round(frame.timestamp - speech_end_time, 2),
            "time_until_end": round(video_duration - frame.timestamp, 2),
        }

    def _build_prompt(
        self, frames_data: List[dict], video_duration: float, speech_end_time: float
    ) -> str:
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

    def _call_openai_vision(
        self, frames_data: List[dict], video_duration: float, speech_end_time: float
    ) -> int:
        prompt = self._build_prompt(frames_data, video_duration, speech_end_time)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        for data in frames_data:
            base64_image = base64.b64encode(data["image_bytes"]).decode("utf-8")
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                }
            )

        response = self.client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=10, temperature=0
        )

        response_text = response.choices[0].message.content.strip()

        try:
            selected_number = int(response_text)
            if 1 <= selected_number <= len(frames_data):
                selected_frame = frames_data[selected_number - 1]["frame"]
                logger.info(
                    "LLM selected frame %d at %.4fs (algorithmic rank: %d)",
                    selected_number,
                    selected_frame.timestamp,
                    selected_frame.rank,
                )
                return selected_number - 1
            else:
                logger.warning(
                    "LLM returned invalid frame number: %d (expected 1-%d), using top algorithmic result",
                    selected_number,
                    len(frames_data),
                )
                return 0
        except ValueError:
            logger.warning(
                "LLM returned non-numeric response: '%s', using top algorithmic result",
                response_text,
            )
            return 0

    async def select_best_frame_async(
        self,
        video_path: str,
        ranked_frames: List[RankedFrame],
        speech_end_time: float,
        video_duration: float,
    ) -> RankedFrame:
        """
        Async version of select_best_frame.

        Select the best frame from top candidates using LLM vision analysis.
        Frame extraction runs in thread pool, OpenAI API call is async.
        """
        if len(ranked_frames) == 0:
            raise ValueError("No frames provided for selection")

        if len(ranked_frames) == 1:
            return ranked_frames[0]

        candidates = ranked_frames[:5]

        """
        Frame extraction is CPU-bound (OpenCV), so we run it in thread pool
        while the async context allows other coroutines to run.
        """
        frames_data = await asyncio.to_thread(
            self._extract_frames_parallel, video_path, candidates, speech_end_time, video_duration
        )

        if not frames_data:
            raise ValueError("No valid frame data for LLM selection")

        selected_index = await self._call_openai_vision_async(
            frames_data, video_duration, speech_end_time
        )
        return frames_data[selected_index]["frame"]

    async def _call_openai_vision_async(
        self, frames_data: List[dict], video_duration: float, speech_end_time: float
    ) -> int:
        """Async version of _call_openai_vision using AsyncOpenAI client."""
        prompt = self._build_prompt(frames_data, video_duration, speech_end_time)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        for data in frames_data:
            base64_image = base64.b64encode(data["image_bytes"]).decode("utf-8")
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                }
            )

        response = await self.async_client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=10, temperature=0
        )

        response_text = response.choices[0].message.content.strip()

        try:
            selected_number = int(response_text)
            if 1 <= selected_number <= len(frames_data):
                selected_frame = frames_data[selected_number - 1]["frame"]
                logger.info(
                    "LLM selected frame %d at %.4fs (algorithmic rank: %d)",
                    selected_number,
                    selected_frame.timestamp,
                    selected_frame.rank,
                )
                return selected_number - 1
            else:
                logger.warning(
                    "LLM returned invalid frame number: %d (expected 1-%d), using top algorithmic result",
                    selected_number,
                    len(frames_data),
                )
                return 0
        except ValueError:
            logger.warning(
                "LLM returned non-numeric response: '%s', using top algorithmic result",
                response_text,
            )
            return 0
