import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from .config import RankingConfig
from .extractors import FeatureExtractor
from .models import FrameFeatures, FrameScore, RankedFrame
from .scorer import FrameScorer


class CutPointRanker:
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.config.validate()

        self.extractor = FeatureExtractor()
        self.scorer = FrameScorer(self.config)

    def rank_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = 1,
        save_frames: bool = False,
        save_video: bool = False
    ) -> List[RankedFrame]:
        """
        Rank frames in the given time range for optimal cut points.
        Uses multi-stage ranking with temporal diversity filtering.

        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_rate: Process every Nth frame (1 = all frames)
            save_frames: If True, save frames with bounding boxes showing tracked features
            save_video: If True, cut video from start to best timestamp and save as <filename>_cut.mp4

        Returns:
            List of RankedFrame sorted by score (best first) with temporal diversity
        """
        features = self._extract_features(video_path, start_time, end_time, sample_rate)

        if not features:
            return []

        scores = self.scorer.compute_scores(features)

        # Sort by final score (best first)
        sorted_scores = sorted(scores, key=lambda x: x.final_score, reverse=True)

        ranked_frames = [
            RankedFrame(
                rank=i + 1,
                frame_index=score.frame_index,
                timestamp=score.timestamp,
                score=score.final_score
            )
            for i, score in enumerate(sorted_scores)
        ]

        if save_frames:
            self._save_ranked_frames(video_path, ranked_frames)

        if save_video and ranked_frames:
            self._save_cut_video(video_path, ranked_frames[0].timestamp)

        return ranked_frames

    def get_detailed_scores(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = 1
    ) -> List[FrameScore]:
        """
        Get detailed scoring breakdown for all frames.
        Useful for debugging and understanding the ranking.
        Includes quality penalties and stability boosts.
        """
        features = self._extract_features(video_path, start_time, end_time, sample_rate)

        if not features:
            return []

        scores = self.scorer.compute_scores(features)
        sorted_scores = sorted(scores, key=lambda x: x.final_score, reverse=True)
        return sorted_scores

    def _extract_features(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int
    ) -> List[FrameFeatures]:
        """
        Pass 1: Extract all features from frames in the time range.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        self.extractor.reset()

        features = []
        current_frame_idx = start_frame

        while current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame_idx - start_frame) % sample_rate == 0:
                timestamp = current_frame_idx / fps

                eye_openness = self.extractor.extract_eye_openness(frame)
                motion_magnitude = self.extractor.extract_motion_magnitude(frame)
                expression_activity = self.extractor.extract_expression_activity(frame)
                pose_deviation = self.extractor.extract_pose_deviation(frame)
                sharpness = self.extractor.extract_visual_sharpness(frame)

                features.append(FrameFeatures(
                    frame_index=current_frame_idx,
                    timestamp=timestamp,
                    eye_openness=eye_openness,
                    motion_magnitude=motion_magnitude,
                    expression_activity=expression_activity,
                    pose_deviation=pose_deviation,
                    sharpness=sharpness
                ))

            current_frame_idx += 1

        cap.release()
        return features

    def _save_ranked_frames(self, video_path: str, ranked_frames: List[RankedFrame]) -> None:
        """
        Save frames with bounding boxes highlighting tracked features.

        Args:
            video_path: Path to video file
            ranked_frames: List of ranked frames to save
        """
        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for ranked_frame in ranked_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ranked_frame.frame_index)
            ret, frame = cap.read()

            if not ret:
                continue

            annotated_frame = self._draw_feature_boxes(frame.copy())

            output_filename = (
                f"rank_{ranked_frame.rank:03d}_"
                f"frame_{ranked_frame.frame_index}_"
                f"timestamp_{ranked_frame.timestamp:.2f}.jpg"
            )
            output_path = output_dir / output_filename

            cv2.imwrite(str(output_path), annotated_frame)

        cap.release()
        print(f"Saved {len(ranked_frames)} annotated frames to: {output_dir}")

    def _draw_feature_boxes(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw landmark dots for all MediaPipe facial features.

        Args:
            frame: Input frame

        Returns:
            Frame with landmark dots drawn
        """
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return frame

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Define landmark groups with colors (BGR format)
        # Eyes
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Eyebrows
        left_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_eyebrow_indices = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

        # Lips/Mouth outer
        lips_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        # Lips/Mouth inner
        lips_inner_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        # Nose
        nose_indices = [1, 2, 98, 327, 289, 279, 331, 294, 358, 439, 438, 457, 459, 460, 326, 2, 98, 97, 2, 326, 370, 94, 19, 1, 4, 5, 6, 168, 188, 122, 6]

        # Face oval
        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # Draw all landmarks with color coding
        for i, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            # Determine color based on landmark group
            if i in left_eye_indices or i in right_eye_indices:
                color = (0, 255, 0)  # Green for eyes
            elif i in left_eyebrow_indices or i in right_eyebrow_indices:
                color = (0, 255, 255)  # Yellow for eyebrows
            elif i in lips_outer_indices or i in lips_inner_indices:
                color = (255, 0, 0)  # Blue for mouth/lips
            elif i in nose_indices:
                color = (255, 0, 255)  # Magenta for nose
            elif i in face_oval_indices:
                color = (255, 255, 0)  # Cyan for face oval
            else:
                color = (200, 200, 200)  # Light gray for other landmarks

            # Draw dot
            cv2.circle(frame, (x, y), 2, color, -1)

        # Add legend
        legend_y = 30
        cv2.putText(frame, "Eyes", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Eyebrows", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "Mouth", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "Nose", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(frame, "Face Oval", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        face_mesh.close()
        return frame

    def _save_cut_video(self, video_path: str, cut_timestamp: float) -> None:
        """
        Cut video from start to the specified timestamp and save it.

        Args:
            video_path: Path to input video file
            cut_timestamp: Timestamp where to cut the video (in seconds)
        """
        import subprocess

        video_base_name = Path(video_path).stem
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{video_base_name}_cut.mp4"
        output_path = output_dir / output_filename

        # Use ffmpeg to cut the video from start to cut_timestamp
        # Re-encode for frame-accurate cutting (slower but accurate)
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-t', str(cut_timestamp),
            '-c:v', 'libx264',  # Re-encode video for frame accuracy
            '-c:a', 'aac',      # Re-encode audio
            '-y',  # Overwrite output file if exists
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Saved cut video (0.00s - {cut_timestamp:.2f}s) to: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error cutting video: {e.stderr}")
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg to use the save_video feature.")
