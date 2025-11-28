# Airtable Integration Guide

SceneFlow now supports uploading analysis results directly to Airtable! This feature automatically creates the required table schema and uploads video files, frame images, and analysis metadata.

## Features

- **Automatic Schema Creation**: Creates the required Airtable table structure if it doesn't exist
- **Direct File Upload**: Uploads raw video, selected frame image, and cut video as attachments
- **Complete Metadata**: Stores detailed analysis data including scores, features, and configuration
- **CLI & API Support**: Available in both command-line and programmatic interfaces

## Setup

### 1. Install Dependencies

The pyairtable library is included in the standard SceneFlow installation:

```bash
uv sync
```

### 2. Get Airtable Credentials

You need two pieces of information from Airtable:

1. **Access Token**:
   - Go to https://airtable.com/create/tokens
   - Create a new token with the following scopes:
     - `data.records:read`
     - `data.records:write`
     - `schema.bases:read`
     - `schema.bases:write`
   - Add access to the specific base you want to use

2. **Base ID**:
   - Open your Airtable base
   - The URL looks like: `https://airtable.com/appXXXXXXXXXXXXXX/...`
   - Copy the part that starts with `app` (e.g., `appXXXXXXXXXXXXXX`)

### 3. Set Environment Variables

```bash
# Required
export AIRTABLE_ACCESS_TOKEN="your_token_here"
export AIRTABLE_BASE_ID="appXXXXXXXXXXXXXX"

# Optional (defaults to "SceneFlow Analysis")
export AIRTABLE_TABLE_NAME="My Custom Table Name"
```

On Windows (Command Prompt):
```cmd
set AIRTABLE_ACCESS_TOKEN=your_token_here
set AIRTABLE_BASE_ID=appXXXXXXXXXXXXXX
set AIRTABLE_TABLE_NAME=My Custom Table Name
```

On Windows (PowerShell):
```powershell
$env:AIRTABLE_ACCESS_TOKEN="your_token_here"
$env:AIRTABLE_BASE_ID="appXXXXXXXXXXXXXX"
$env:AIRTABLE_TABLE_NAME="My Custom Table Name"
```

## Usage

### CLI Usage

Add the `--airtable` flag to any SceneFlow command:

```bash
# Basic usage
sceneflow video.mp4 --airtable

# With verbose output
sceneflow video.mp4 --airtable --verbose

# With top N results
sceneflow video.mp4 --airtable --top-n 5 --verbose

# From URL
sceneflow https://example.com/video.mp4 --airtable
```

### API Usage

Use the `upload_to_airtable` parameter in API functions:

```python
from sceneflow import get_cut_frame, get_ranked_cut_frames

# Upload single best cut point
timestamp = get_cut_frame(
    "video.mp4",
    upload_to_airtable=True
)

# Upload with custom credentials (overrides environment variables)
timestamp = get_cut_frame(
    "video.mp4",
    upload_to_airtable=True,
    airtable_access_token="your_token",
    airtable_base_id="appXXXXXXXXXXXXXX",
    airtable_table_name="Custom Table"
)

# Get top N cut points and upload best one
timestamps = get_ranked_cut_frames(
    "video.mp4",
    n=5,
    upload_to_airtable=True
)
```

### Direct Uploader Usage

For more control, use the `AirtableUploader` class:

```python
from sceneflow.airtable_uploader import AirtableUploader
from sceneflow import CutPointRanker
from sceneflow.speech_detector import SpeechDetector

# Initialize uploader
uploader = AirtableUploader()
uploader.ensure_table_schema()

# Perform analysis
detector = SpeechDetector()
speech_end = detector.get_speech_end_time("video.mp4")

ranker = CutPointRanker()
ranked_frames = ranker.rank_frames("video.mp4", speech_end, duration=10.0)

# Get detailed data
scores = ranker.get_detailed_scores("video.mp4", speech_end, 10.0)
features = ranker._extract_features("video.mp4", speech_end, 10.0, sample_rate=2)

# Upload to Airtable
record_id = uploader.upload_analysis(
    video_path="video.mp4",
    best_frame=ranked_frames[0],
    frame_score=scores[0],
    frame_features=features[0],
    speech_end_time=speech_end,
    duration=10.0,
    config_dict={"sample_rate": 2, "weights": {...}}
)

print(f"Uploaded to Airtable! Record ID: {record_id}")
```

## Airtable Schema

The integration automatically creates a table with the following fields:

| Field Name | Type | Description |
|------------|------|-------------|
| **Timestamp** *(Primary)* | Single Line Text | Cut point timestamp (e.g., "8.45s") |
| **Raw Video** | Multiple Attachments | Original input video file |
| **Selected Frame Image** | Multiple Attachments | JPEG image of the best-ranked frame |
| **Output Video** | Multiple Attachments | Cut video from start to optimal timestamp |
| **Raw Data** | Long Text | Complete JSON with analysis details |

> **Note**: "Timestamp" is the primary field because Airtable doesn't allow attachment fields as primary fields.

### Raw Data JSON Structure

The "Raw Data" field contains a comprehensive JSON object:

```json
{
  "rank": 1,
  "frame_index": 1234,
  "timestamp": "8.45",
  "score": 0.8756,
  "video_info": {
    "source_filename": "video.mp4",
    "duration_seconds": 10.5,
    "speech_end_time": 8.45,
    "analysis_range": "8.45s - 10.50s"
  },
  "score_breakdown": {
    "composite_score": 0.85,
    "context_score": 0.87,
    "quality_penalty": 0.95,
    "stability_boost": 1.08,
    "component_scores": {
      "eye_openness": 0.82,
      "motion_stability": 0.91,
      "expression_neutrality": 0.88,
      "pose_stability": 0.79,
      "visual_sharpness": 0.87
    }
  },
  "raw_features": {
    "eye_openness": 0.28,
    "motion_magnitude": 2.3,
    "expression_activity": 0.12,
    "pose_deviation": 3.5,
    "sharpness": 145.2,
    "num_faces": 1
  },
  "config": {
    "sample_rate": 2,
    "weights": {
      "eye_openness": 0.30,
      "motion_stability": 0.25,
      "expression_neutrality": 0.20,
      "pose_stability": 0.15,
      "visual_sharpness": 0.10
    }
  }
}
```

## Requirements

- **ffmpeg**: Required for generating cut videos (same requirement as `--save-video` flag)
- **pyairtable >= 3.2.0**: Automatically installed with SceneFlow
- **Airtable Access Token**: Must have read/write permissions for data and schema

## Error Handling

The integration includes comprehensive error handling:

- **Missing Credentials**: Clear error messages guide you to set environment variables
- **Table Creation**: Automatically creates table if it doesn't exist
- **Upload Failures**: Detailed error messages for troubleshooting
- **Non-blocking**: Analysis completes even if Airtable upload fails (error is logged)

## Common Issues

### "pyairtable is not installed"
```bash
uv sync  # or pip install pyairtable>=3.2.0
```

### "Airtable access token is required"
Make sure you've set the `AIRTABLE_ACCESS_TOKEN` environment variable.

### "ffmpeg not found"
Install ffmpeg from https://ffmpeg.org/download.html and ensure it's in your PATH.

### "Failed to create table"
Check that your access token has `schema.bases:write` permission.

## Testing

Run the test suite to verify the integration:

```bash
uv run python test_airtable_integration.py
```

This tests:
- Import functionality
- Credential validation
- CLI flag presence
- API parameter availability

## Examples

### Full Workflow Example

```bash
# Set credentials
export AIRTABLE_ACCESS_TOKEN="patXXXXXXXXXXXXXX"
export AIRTABLE_BASE_ID="appXXXXXXXXXXXXXX"

# Run analysis with Airtable upload
sceneflow my_video.mp4 --airtable --verbose

# Output shows:
# - Speech detection progress
# - Frame analysis progress
# - Airtable upload confirmation
# - Record ID for reference
```

### Batch Processing Example

```python
from sceneflow import get_cut_frame
from pathlib import Path

videos = Path("videos/").glob("*.mp4")

for video in videos:
    try:
        timestamp = get_cut_frame(
            str(video),
            upload_to_airtable=True,
            sample_rate=2
        )
        print(f"✓ {video.name}: {timestamp:.2f}s (uploaded)")
    except Exception as e:
        print(f"✗ {video.name}: {e}")
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/vertexcover-io/sceneflow/issues
- Documentation: https://github.com/vertexcover-io/sceneflow#readme
