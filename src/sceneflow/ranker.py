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

        self.extractor = FeatureExtractor(
            center_weighting_strength=self.config.center_weighting_strength,
            min_face_confidence=self.config.min_face_confidence
        )
        self.scorer = FrameScorer(self.config)

    def rank_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = 1,
        save_frames: bool = False,
        save_video: bool = False,
        output_path: Optional[str] = None,
        save_logs: bool = False,
        return_internals: bool = False
    ):
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
            output_path: Optional custom path for saved video. If None, uses output/<filename>_cut.mp4
            save_logs: If True, save detailed analysis data to JSONL files for each frame
            return_internals: If True, return tuple of (ranked_frames, features, scores) to avoid re-processing

        Returns:
            If return_internals=False: List of RankedFrame sorted by score (best first)
            If return_internals=True: Tuple of (List[RankedFrame], List[FrameFeatures], List[FrameScore])
        """
        features = self._extract_features(video_path, start_time, end_time, sample_rate)

        if not features:
            if return_internals:
                return [], [], []
            return []

        scores = self.scorer.compute_scores(features)

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

        if save_logs:
            self._save_frame_logs(video_path, ranked_frames, features, scores)

        if save_video and ranked_frames:
            self._save_cut_video(video_path, ranked_frames[0].timestamp, output_path=output_path)

        if return_internals:
            return ranked_frames, features, scores
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

        print(f"start frame : {start_frame} , end frame : {end_frame}")

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

                # Extract all faces and get aggregated metrics
                num_faces, face_features_list, (eye_openness, expression_activity, pose_deviation) = \
                    self.extractor.extract_all_faces(frame)

                # Extract frame-level metrics (not face-specific)
                motion_magnitude = self.extractor.extract_motion_magnitude(frame)
                sharpness = self.extractor.extract_visual_sharpness(frame)

                features.append(FrameFeatures(
                    frame_index=current_frame_idx,
                    timestamp=timestamp,
                    eye_openness=eye_openness,
                    motion_magnitude=motion_magnitude,
                    expression_activity=expression_activity,
                    pose_deviation=pose_deviation,
                    sharpness=sharpness,
                    num_faces=num_faces,
                    individual_faces=face_features_list if face_features_list else None
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

    def _save_frame_logs(
        self,
        video_path: str,
        ranked_frames: List[RankedFrame],
        features: List[FrameFeatures],
        scores: List[FrameScore]
    ) -> None:
        import json

        video_base_name = Path(video_path).stem
        output_dir = Path("output") / video_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_to_features = {f.frame_index: f for f in features}
        frame_to_scores = {s.frame_index: s for s in scores}

        for ranked_frame in ranked_frames:
            frame_idx = ranked_frame.frame_index
            feature = frame_to_features.get(frame_idx)
            score = frame_to_scores.get(frame_idx)

            if not feature or not score:
                continue

            log_data = {
                "metadata": {
                    "rank": ranked_frame.rank,
                    "frame_index": ranked_frame.frame_index,
                    "timestamp": ranked_frame.timestamp
                },
                "raw_features": {
                    "eye_openness": feature.eye_openness,
                    "motion_magnitude": feature.motion_magnitude,
                    "expression_activity": feature.expression_activity,
                    "pose_deviation": feature.pose_deviation,
                    "sharpness": feature.sharpness,
                    "num_faces": feature.num_faces
                },
                "individual_faces": [],
                "normalized_scores": {
                    "eye_openness_score": score.eye_openness_score,
                    "motion_stability_score": score.motion_stability_score,
                    "expression_neutrality_score": score.expression_neutrality_score,
                    "pose_stability_score": score.pose_stability_score,
                    "visual_sharpness_score": score.visual_sharpness_score
                },
                "score_breakdown": {
                    "composite_score": score.composite_score,
                    "context_score": score.context_score,
                    "quality_penalty": score.quality_penalty,
                    "stability_boost": score.stability_boost,
                    "final_score": score.final_score
                },
                "configuration": {
                    "eye_openness_weight": self.config.eye_openness_weight,
                    "motion_stability_weight": self.config.motion_stability_weight,
                    "expression_neutrality_weight": self.config.expression_neutrality_weight,
                    "pose_stability_weight": self.config.pose_stability_weight,
                    "visual_sharpness_weight": self.config.visual_sharpness_weight,
                    "context_window_size": self.config.context_window_size,
                    "quality_gate_percentile": self.config.quality_gate_percentile,
                    "local_stability_window": self.config.local_stability_window,
                    "center_weighting_strength": self.config.center_weighting_strength,
                    "min_face_confidence": self.config.min_face_confidence
                }
            }

            if feature.individual_faces:
                for face in feature.individual_faces:
                    log_data["individual_faces"].append({
                        "face_index": face.face_index,
                        "bbox": face.bbox,
                        "center_distance": face.center_distance,
                        "center_weight": face.center_weight,
                        "eye_openness": face.eye_openness,
                        "expression_activity": face.expression_activity,
                        "pose_deviation": face.pose_deviation
                    })

            output_filename = (
                f"rank_{ranked_frame.rank:03d}_"
                f"frame_{ranked_frame.frame_index}_"
                f"timestamp_{ranked_frame.timestamp:.2f}.jsonl"
            )
            output_path = output_dir / output_filename

            with open(output_path, 'w') as f:
                json.dump(log_data, f, indent=2)

        print(f"Saved {len(ranked_frames)} log files to: {output_dir}")

    def _draw_feature_boxes(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw landmark dots for all InsightFace 106 facial features for ALL detected faces.

        Args:
            frame: Input frame

        Returns:
            Frame with landmark dots drawn for all faces
        """
        from .extractors import (
            LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_OUTER_INDICES
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.extractor.app.get(rgb_frame)

        if not faces or len(faces) == 0:
            return frame

        # Draw features for ALL detected faces
        for face_idx, face in enumerate(faces):
            # Draw bounding box
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Add face number label
            label = f"Face {face_idx + 1}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if 106 landmarks are available
            if self.extractor.has_106_landmarks and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                landmarks = face.landmark_2d_106.astype(int)

                # Draw all landmarks with color coding (BGR format)
                for i, point in enumerate(landmarks):
                    # Determine color based on landmark group
                    if i in LEFT_EYE_INDICES or i in RIGHT_EYE_INDICES:
                        color = (255, 0, 0)  # Blue for eyes
                    elif i in MOUTH_OUTER_INDICES:
                        color = (0, 0, 255)  # Red for mouth
                    else:
                        color = (200, 160, 75)  # Beige for other landmarks

                    # Draw dot
                    cv2.circle(frame, tuple(point), 1, color, -1)
            else:
                # Fallback: draw 5-point landmarks if available
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int)
                    landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
                    colors = [
                        (255, 0, 0),    # Left eye - Blue
                        (255, 0, 0),    # Right eye - Blue
                        (0, 255, 255),  # Nose - Yellow
                        (0, 0, 255),    # Left mouth - Red
                        (0, 0, 255)     # Right mouth - Red
                    ]

                    for point, name, color in zip(landmarks, landmark_names, colors):
                        cv2.circle(frame, (point[0], point[1]), 3, color, -1)

        # Add legend (only once for all faces)
        legend_y = 30
        cv2.putText(frame, f"Faces: {len(faces)}", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Eyes (Blue)", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "Mouth (Red)", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "Other (Beige)", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 160, 75), 1)

        return frame

    def _save_cut_video(self, video_path: str, cut_timestamp: float, output_path: Optional[str] = None) -> str:
        """
        Cut video from start to the specified timestamp and save it.

        Args:
            video_path: Path to input video file
            cut_timestamp: Timestamp where to cut the video (in seconds)
            output_path: Optional custom output path for the cut video.
                        If None, saves to output/<video_name>_cut.mp4

        Returns:
            Path to the saved cut video
        """
        import subprocess

        if output_path:
            # Use custom output path
            final_output_path = Path(output_path)
            # Create parent directory if it doesn't exist
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default behavior: save to output directory
            video_base_name = Path(video_path).stem
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{video_base_name}_cut.mp4"
            final_output_path = output_dir / output_filename

        # Use ffmpeg to cut the video from start to cut_timestamp
        # Re-encode for frame-accurate cutting (slower but accurate)
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-t', str(cut_timestamp),
            '-c:v', 'libx264',  # Re-encode video for frame accuracy
            '-c:a', 'aac',      # Re-encode audio
            '-y',  # Overwrite output file if exists
            str(final_output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Saved cut video (0.00s - {cut_timestamp:.2f}s) to: {final_output_path}")
            return str(final_output_path)
        except subprocess.CalledProcessError as e:
            print(f"Error cutting video: {e.stderr}")
            raise
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg to use the save_video feature.")
            raise
