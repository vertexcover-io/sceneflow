# SceneFlow

Smart cut point detection for AI-generated talking head videos using advanced speech detection and multi-factor visual analysis.

## Overview

SceneFlow automatically identifies the optimal point to cut AI-generated talking head videos by analyzing speech patterns and visual features. It eliminates awkward mid-speech cuts and unwanted motion at the end of videos.

## Key Features

- **Speech Detection** - Silero VAD for precise speech boundary identification
- **Energy-Based Refinement** - Frame-accurate speech end detection using audio energy analysis
- **Visual Analysis** - Multi-factor ranking based on eye openness, motion stability, expression neutrality, pose stability, and visual sharpness
- **Multi-Face Support** - Analyzes all faces with center-weighted scoring
- **LLM Integration** - Optional GPT-4o vision for selecting the best frame
- **URL Support** - Direct video URL processing via HTTP GET
- **Airtable Integration** - Upload results and videos for tracking
- **InsightFace-Powered** - 106-landmark facial detection for precise analysis

## Installation

```bash
pip install sceneflow
```

### Requirements

- Python 3.9 or higher
- FFmpeg (for video processing)
- OpenAI API key (optional, for LLM-powered frame selection)

## Quick Start

### Command Line Interface

**Basic Usage**

```bash
# Get cut timestamp
sceneflow video.mp4

# Verbose output with detailed analysis
sceneflow video.mp4 --verbose

# Get top 5 best cut points
sceneflow video.mp4 --top-n 5

# Process video from URL
sceneflow "https://example.com/video.mp4" --verbose
```

**Advanced Options**

```bash
# Save outputs (frames, video, logs)
sceneflow video.mp4 --save-frames --save-video --save-logs

# Custom output path for video
sceneflow video.mp4 --output /path/to/output.mp4

# Use LLM for frame selection
sceneflow video.mp4 --use-llm-selection

# Upload results to Airtable
sceneflow video.mp4 --airtable

# Disable energy refinement
sceneflow video.mp4 --no-energy-refinement

# Adjust energy refinement parameters
sceneflow video.mp4 --energy-threshold-db 10.0 --energy-lookback-frames 25

# Disable visual analysis (faster, speech detection only)
sceneflow video.mp4 --disable-visual-analysis
```

### Python API

**Simple API**

```python
from sceneflow import get_cut_frame, get_ranked_cut_frames

# Get the best cut point
best_time = get_cut_frame("video.mp4")
print(f"Cut at: {best_time:.2f}s")

# Get top 5 cut points
top_5 = get_ranked_cut_frames("video.mp4", n=5)
for i, time in enumerate(top_5, 1):
    print(f"{i}. {time:.2f}s")

# With LLM-powered frame selection
best_time = get_cut_frame(
    "video.mp4",
    use_llm_selection=True,
    openai_api_key="your-api-key"
)

# Upload to Airtable
best_time = get_cut_frame(
    "video.mp4",
    upload_to_airtable=True,
    save_video=True
)

# Disable energy refinement
best_time = get_cut_frame(
    "video.mp4",
    use_energy_refinement=False
)
```

**Advanced API**

```python
from sceneflow import CutPointRanker
from sceneflow.speech_detector import SpeechDetector
import cv2

# Detect speech end time
detector = SpeechDetector()
speech_end_time, confidence = detector.get_speech_end_time(
    "video.mp4",
    return_confidence=True
)

# Get video duration
cap = cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
cap.release()

# Rank frames after speech ends
ranker = CutPointRanker()
ranked_frames = ranker.rank_frames(
    video_path="video.mp4",
    start_time=speech_end_time,
    end_time=duration,
    sample_rate=2
)

# Get best cut point
best_cut = ranked_frames[0]
print(f"Best cut point: {best_cut.timestamp:.2f}s (score: {best_cut.score:.4f})")
```

## API Reference

### Common Parameters

| Parameter | Type | Default | Used By | Description |
|-----------|------|---------|---------|-------------|
| `source` | str | required | All | Video file path or URL |
| `output_path` | str | required | cut_video | Output path for cut video |
| `n` | int | 5 | get_ranked | Number of timestamps to return |
| `sample_rate` | int | 2 | All | Process every Nth frame |
| `save_video` | bool | False | get_cut_frame | Save cut video |
| `save_frames` | bool | False | All | Save annotated frames |
| `save_logs` | bool | False | All | Save analysis logs |
| `upload_to_airtable` | bool | False | All | Upload to Airtable |
| `use_llm_selection` | bool | False | All | Use GPT-4o for selection |
| `use_energy_refinement` | bool | True | All | Refine VAD with energy analysis |
| `energy_threshold_db` | float | 8.0 | All | Minimum dB drop for refinement |
| `energy_lookback_frames` | int | 20 | All | Max frames to search backward |
| `disable_visual_analysis` | bool | False | All | Skip visual ranking, use speech end only |
| `openai_api_key` | str | None | All | OpenAI API key (or use env var) |
| `airtable_access_token` | str | None | All | Airtable token (or use env var) |
| `airtable_base_id` | str | None | All | Airtable base ID (or use env var) |
| `airtable_table_name` | str | None | All | Airtable table name (or use env var) |

**Functions:**
- `get_cut_frame(source, **params)` - Returns best cut timestamp (float)
- `get_ranked_cut_frames(source, n=5, **params)` - Returns top N timestamps (list)
- `cut_video(source, output_path, **params)` - Cuts video and returns timestamp (float)

## CLI Reference

```
sceneflow SOURCE [OPTIONS]

Arguments:
  SOURCE                         Path to video file or URL

Output Options:
  --verbose                      Show detailed analysis
  --json-output PATH             Save analysis to JSON (directory path)
  --top-n INT                    Return top N timestamps with scores

Processing Options:
  --sample-rate INT              Process every Nth frame (default: 2)
  --save-frames                  Save annotated frames with landmarks
  --save-video                   Save cut video
  --output PATH                  Custom output path for video
  --save-logs                    Save detailed logs

Speech Detection Options:
  --no-energy-refinement         Disable energy-based refinement
  --energy-threshold-db FLOAT    Minimum dB drop (default: 8.0)
  --energy-lookback-frames INT   Max frames to search backward (default: 20)

Visual Analysis Options:
  --disable-visual-analysis      Disable visual analysis and return speech end time only (faster)

Advanced Options:
  --use-llm-selection            Use GPT-4o for frame selection
  --airtable                     Upload results to Airtable

  --help                         Show help message
  --version                      Show version
```

## Environment Variables

For optional features, set these environment variables:

```bash
# Airtable Integration
AIRTABLE_ACCESS_TOKEN     # Your Airtable access token
AIRTABLE_BASE_ID          # Your Airtable base ID
AIRTABLE_TABLE_NAME       # Table name (optional, defaults to "SceneFlow Analysis")

# LLM-Powered Selection
OPENAI_API_KEY            # OpenAI API key for GPT-4o integration
```

Quick setup:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## How It Works

### Speech Detection

Uses Silero VAD (Voice Activity Detection) for accurate speech/silence detection. Energy-based refinement analyzes audio energy around the VAD timestamp to find the exact frame where speech ends, typically adjusting by 3-5 frames for frame-accurate boundaries.

### Visual Analysis

After identifying speech end, SceneFlow analyzes frames using InsightFace (106-landmark facial detection) and ranks them based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| Expression Neutrality | 30% | Calm, neutral facial expressions |
| Motion Stability | 25% | Minimal optical flow between frames |
| Eye Openness | 20% | Natural eye openness |
| Pose Stability | 15% | Steady head position |
| Visual Sharpness | 10% | Clear frame quality |

When multiple faces are detected, center-weighted averaging prioritizes faces closer to the frame center.

### Ranking System

1. Extract raw features from all frames
2. Normalize metrics across the entire frame set
3. Apply weighted scoring based on configuration
4. Use temporal context windows for stable sequences
5. Return ranked list of candidates

## Examples

Check the `examples/` directory:
- `basic_usage.py` - Simple API usage
- `custom_config.py` - Custom configuration
- `ranked_results.py` - Multiple cut points
- `url_download.py` - Working with URLs
- `batch_processing.py` - Processing multiple videos
- `save_logs_example.py` - Detailed analysis logs

## Technical Details

- **Speech Detection**: Silero VAD
- **Energy Refinement**: Audio energy analysis with dB-based drop detection
- **Facial Analysis**: InsightFace with 106-landmark detection
- **Multi-Face Support**: Center-weighted averaging
- **Motion Analysis**: Farneback optical flow
- **Eye Detection**: Eye Aspect Ratio (EAR) using 7-point eye landmarks
- **Expression Analysis**: Mouth Aspect Ratio (MAR) using 20-point mouth landmarks
- **Frame Quality**: Laplacian variance
- **Ranking**: Multi-factor scoring with temporal context windows
- **LLM Selection**: GPT-4o vision analysis

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- Repository: https://github.com/vertexcover-io/sceneflow
- Issues: https://github.com/vertexcover-io/sceneflow/issues
- PyPI: https://pypi.org/project/sceneflow/

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
