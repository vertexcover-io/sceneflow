"""Airtable integration for uploading SceneFlow analysis results."""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from pyairtable import Api, Base, Table
    PYAIRTABLE_AVAILABLE = True
except ImportError:
    PYAIRTABLE_AVAILABLE = False

from sceneflow.shared.constants import VIDEO
from sceneflow.shared.models import RankedFrame, FrameScore, FrameFeatures
from sceneflow.utils.video import extract_frame, cut_video_to_bytes

logger = logging.getLogger(__name__)


class AirtableUploader:
    """
    Handles uploading SceneFlow analysis results to Airtable.

    Automatically creates the required table schema if it doesn't exist.
    Uploads raw video, selected frame image, and cut video as attachments.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        base_id: Optional[str] = None,
        table_name: Optional[str] = None
    ):
        """
        Initialize Airtable uploader.

        Args:
            access_token: Airtable access token (defaults to AIRTABLE_ACCESS_TOKEN env var)
            base_id: Airtable base ID (defaults to AIRTABLE_BASE_ID env var)
            table_name: Table name (defaults to AIRTABLE_TABLE_NAME or "SceneFlow Analysis")

        Raises:
            RuntimeError: If pyairtable is not installed or credentials are missing
        """
        if not PYAIRTABLE_AVAILABLE:
            raise RuntimeError(
                "pyairtable is not installed. Install it with: pip install pyairtable>=3.2.0"
            )

        # Get credentials from parameters or environment variables
        self.access_token = access_token or os.environ.get("AIRTABLE_ACCESS_TOKEN")
        self.base_id = base_id or os.environ.get("AIRTABLE_BASE_ID")
        self.table_name = table_name or os.environ.get("AIRTABLE_TABLE_NAME", "SceneFlow Analysis")

        if not self.access_token:
            raise RuntimeError(
                "Airtable access token is required. Set AIRTABLE_ACCESS_TOKEN environment variable "
                "or pass access_token parameter."
            )

        if not self.base_id:
            raise RuntimeError(
                "Airtable base ID is required. Set AIRTABLE_BASE_ID environment variable "
                "or pass base_id parameter."
            )

        # Initialize Airtable API
        self.api = Api(self.access_token)
        self.base = self.api.base(self.base_id)
        self.table: Optional[Table] = None

        logger.info(f"Initialized Airtable uploader for base: {self.base_id}, table: {self.table_name}")

    def ensure_table_schema(self) -> None:
        """
        Ensure the table exists with the correct schema.
        Creates the table if it doesn't exist.

        Required fields:
        - Raw Video (multipleAttachments)
        - Selected Frame Image (multipleAttachments)
        - Output Video (multipleAttachments)
        - Timestamp (singleLineText)
        - Raw Data (multilineText)
        """
        try:
            # Check if table exists
            schema = self.base.schema()
            existing_table = None

            try:
                existing_table = schema.table(self.table_name)
                logger.info(f"Found existing table: {self.table_name}")
            except KeyError:
                logger.info(f"Table '{self.table_name}' does not exist, will create it")

            if existing_table:
                # Table exists, use it
                self.table = self.base.table(self.table_name)
                logger.info(f"Using existing table: {self.table_name}")
            else:
                # Create new table with required schema
                logger.info(f"Creating new table: {self.table_name}")

                # NOTE: First field becomes the primary field
                # Airtable doesn't allow attachment fields as primary, so Timestamp goes first
                fields = [
                    {
                        "name": "Timestamp",
                        "type": "singleLineText",
                        "description": "Cut point timestamp in seconds"
                    },
                    {
                        "name": "Raw Video",
                        "type": "multipleAttachments",
                        "description": "Original input video file"
                    },
                    {
                        "name": "Selected Frame Image",
                        "type": "multipleAttachments",
                        "description": "JPEG image of the best-ranked frame"
                    },
                    {
                        "name": "Output Video",
                        "type": "multipleAttachments",
                        "description": "Cut video from start to optimal timestamp"
                    },
                    {
                        "name": "Raw Data",
                        "type": "multilineText",
                        "description": "Complete JSON with analysis details"
                    }
                ]

                self.table = self.base.create_table(
                    name=self.table_name,
                    fields=fields,
                    description="SceneFlow video analysis results with cut point detection"
                )

                logger.info(f"Successfully created table: {self.table_name}")

        except Exception as e:
            logger.error(f"Failed to ensure table schema: {str(e)}")
            raise RuntimeError(f"Failed to setup Airtable table: {str(e)}")

    def upload_analysis(
        self,
        video_path: str,
        best_frame: RankedFrame,
        frame_score: FrameScore,
        frame_features: FrameFeatures,
        speech_end_time: float,
        duration: float,
        config_dict: Dict[str, Any]
    ) -> str:
        """
        Upload complete analysis results to Airtable.

        Args:
            video_path: Path to the original video file
            best_frame: Best ranked frame result
            frame_score: Detailed score breakdown
            frame_features: Raw feature measurements
            speech_end_time: When speech ended in the video
            duration: Total video duration
            config_dict: Configuration used for analysis

        Returns:
            Airtable record ID

        Raises:
            RuntimeError: If upload fails
        """
        # Ensure table exists
        if not self.table:
            self.ensure_table_schema()

        try:
            logger.info("Starting Airtable upload...")

            # Step 1: Extract frame image
            logger.info("Extracting best frame as image...")
            frame_image_bytes = self._extract_frame_image(video_path, best_frame.frame_index)

            # Step 2: Generate cut video
            logger.info("Generating cut video...")
            cut_video_bytes = self._generate_cut_video(video_path, best_frame.timestamp)

            # Step 3: Read original video
            logger.info("Reading original video...")
            with open(video_path, 'rb') as f:
                video_bytes = f.read()

            # Step 4: Prepare metadata JSON
            metadata = self._prepare_metadata(
                video_path, best_frame, frame_score, frame_features,
                speech_end_time, duration, config_dict
            )
            metadata_json = json.dumps(metadata, indent=2)

            # Step 5: Create record with timestamp and metadata
            logger.info("Creating Airtable record...")
            record = self.table.create({
                "Timestamp": f"{best_frame.timestamp:.2f}s",
                "Raw Data": metadata_json
            })
            record_id = record["id"]
            logger.info(f"Created record: {record_id}")

            # Step 6: Upload attachments
            video_filename = Path(video_path).name
            frame_filename = f"frame_{best_frame.frame_index}_at_{best_frame.timestamp:.2f}s.jpg"
            cut_video_filename = f"{Path(video_path).stem}_cut.mp4"

            logger.info("Uploading raw video...")
            self.table.upload_attachment(
                record_id,
                "Raw Video",
                video_filename,
                video_bytes,
                "video/mp4"
            )

            logger.info("Uploading frame image...")
            self.table.upload_attachment(
                record_id,
                "Selected Frame Image",
                frame_filename,
                frame_image_bytes,
                "image/jpeg"
            )

            logger.info("Uploading cut video...")
            self.table.upload_attachment(
                record_id,
                "Output Video",
                cut_video_filename,
                cut_video_bytes,
                "video/mp4"
            )

            logger.info(f"Successfully uploaded analysis to Airtable! Record ID: {record_id}")
            return record_id

        except Exception as e:
            logger.error(f"Failed to upload to Airtable: {str(e)}")
            raise RuntimeError(f"Airtable upload failed: {str(e)}")

    def _extract_frame_image(self, video_path: str, frame_index: int) -> bytes:
        """
        Extract a specific frame from video as JPEG bytes.

        Args:
            video_path: Path to video file
            frame_index: Frame number to extract

        Returns:
            JPEG image as bytes
        """
        return extract_frame(video_path, frame_index, VIDEO.JPEG_QUALITY_HIGH)

    def _generate_cut_video(self, video_path: str, cut_timestamp: float) -> bytes:
        """
        Generate cut video from start to timestamp using ffmpeg.

        Args:
            video_path: Path to original video
            cut_timestamp: Timestamp to cut at (in seconds)

        Returns:
            Cut video as bytes

        Raises:
            RuntimeError: If ffmpeg is not available or video generation fails
        """
        try:
            return cut_video_to_bytes(video_path, cut_timestamp)
        except Exception as e:
            raise RuntimeError(f"Failed to cut video: {e}")

    def _prepare_metadata(
        self,
        video_path: str,
        best_frame: RankedFrame,
        frame_score: FrameScore,
        frame_features: FrameFeatures,
        speech_end_time: float,
        duration: float,
        config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare complete metadata JSON for the Raw Data field.

        Returns:
            Dictionary containing all analysis metadata
        """
        return {
            "rank": best_frame.rank,
            "frame_index": best_frame.frame_index,
            "timestamp": f"{best_frame.timestamp:.2f}",
            "score": round(best_frame.score, 4),
            "video_info": {
                "source_filename": Path(video_path).name,
                "duration_seconds": round(duration, 2),
                "speech_end_time": round(speech_end_time, 2),
                "analysis_range": f"{speech_end_time:.2f}s - {duration:.2f}s"
            },
            "score_breakdown": {
                "composite_score": round(frame_score.composite_score, 4),
                "context_score": round(frame_score.context_score, 4),
                "quality_penalty": round(frame_score.quality_penalty, 4),
                "stability_boost": round(frame_score.stability_boost, 4),
                "component_scores": {
                    "eye_openness": round(frame_score.eye_openness_score, 4),
                    "motion_stability": round(frame_score.motion_stability_score, 4),
                    "expression_neutrality": round(frame_score.expression_neutrality_score, 4),
                    "pose_stability": round(frame_score.pose_stability_score, 4),
                    "visual_sharpness": round(frame_score.visual_sharpness_score, 4)
                }
            },
            "raw_features": {
                "eye_openness": round(frame_features.eye_openness, 4),
                "motion_magnitude": round(frame_features.motion_magnitude, 4),
                "expression_activity": round(frame_features.expression_activity, 4),
                "pose_deviation": round(frame_features.pose_deviation, 4),
                "sharpness": round(frame_features.sharpness, 2),
                "num_faces": frame_features.num_faces
            },
            "config": config_dict
        }


def upload_to_airtable(
    video_path: str,
    best_frame: RankedFrame,
    frame_score: FrameScore,
    frame_features: FrameFeatures,
    speech_end_time: float,
    duration: float,
    config_dict: Dict[str, Any],
    access_token: Optional[str] = None,
    base_id: Optional[str] = None,
    table_name: Optional[str] = None
) -> str:
    """
    Convenience function to upload analysis results to Airtable.

    Args:
        video_path: Path to the original video file
        best_frame: Best ranked frame result
        frame_score: Detailed score breakdown
        frame_features: Raw feature measurements
        speech_end_time: When speech ended in the video
        duration: Total video duration
        config_dict: Configuration used for analysis
        access_token: Optional Airtable access token (defaults to env var)
        base_id: Optional Airtable base ID (defaults to env var)
        table_name: Optional table name (defaults to env var or "SceneFlow Analysis")

    Returns:
        Airtable record ID

    Raises:
        RuntimeError: If upload fails or credentials are missing
    """
    uploader = AirtableUploader(access_token, base_id, table_name)
    uploader.ensure_table_schema()
    return uploader.upload_analysis(
        video_path,
        best_frame,
        frame_score,
        frame_features,
        speech_end_time,
        duration,
        config_dict
    )
