# SceneFlow

**Smart cut point detection for AI-generated talking head videos**

SceneFlow automatically finds the optimal point to cut your AI-generated talking head videos using advanced speech detection and multi-factor visual analysis. No more awkward mid-speech cuts or unwanted motion at the end of your videos!

## Features

- **Intelligent Speech Detection** - Uses Silero VAD (Voice Activity Detection) to precisely identify when speech ends
- **Energy-Based Refinement** - Frame-accurate speech boundary detection using audio energy analysis
- **Multi-Factor Visual Analysis** - Ranks potential cut points based on:
  - Eye openness (natural blink detection using 7-point eye landmarks)
  - Motion stability (minimal movement via optical flow)
  - Expression neutrality (calm facial expressions using 20-point mouth landmarks)
  - Pose stability (steady head position)
  - Visual sharpness (frame quality)
- **Multi-Face Support** - Automatically detects and analyzes all faces with center-weighted scoring
- **LLM-Powered Selection** *(Optional)* - Uses GPT-4o vision to select the best frame from top algorithmic candidates
- **URL Support** - Works directly with video URLs via HTTP GET
- **Flexible Output** - Simple timestamp output, verbose mode, or detailed JSON analysis
- **Airtable Integration** - Upload results and videos directly to Airtable for tracking
- **Customizable Weights** - Adjust the importance of each visual factor
- **Pure Ranking Algorithm** - No arbitrary thresholds, everything is relative
- **InsightFace-Powered** - Uses advanced 106-landmark facial detection for precise analysis

## What's New

### 🎯 Energy-Based Refinement (v0.2.0+)
Frame-accurate speech boundary detection using audio energy analysis. Refines VAD timestamps by detecting sharp energy drops (8+ dB), typically adjusting by 3-5 frames for perfect precision. **Enabled by default.**

### 👥 Multi-Face Support (v0.2.0+)
Automatically detects and analyzes all faces in each frame using center-weighted averaging. Faces closer to the frame center have higher influence while still considering all detected faces.

### 🤖 LLM-Powered Selection (v0.2.0+)
Optional GPT-4o vision integration that analyzes the top 5 algorithmic candidates to select the best frame based on natural expressions, appropriate padding, and visual quality.

### 📊 Airtable Integration (v0.2.0+)
Upload analysis results, videos, and frame images directly to Airtable with automatic schema creation. Perfect for tracking and reviewing results.

## Installation

```bash
pip install sceneflow
```

### Requirements

- Python 3.9 or higher
- FFmpeg (for video processing and cut video generation)
- **Optional**: OpenAI API key (for LLM-powered frame selection)

## Quick Start

### Command Line

```bash
# Basic usage - outputs just the timestamp
sceneflow video.mp4
# Output: 5.23

# Verbose mode - see detailed analysis
sceneflow video.mp4 --verbose

# Get top 5 best cut points with scores
sceneflow video.mp4 --top-n 5

# Save detailed JSON analysis
sceneflow video.mp4 --json-output ./output

# From a URL
sceneflow "https://example.com/video.mp4" --verbose

# Save outputs (frames, video, logs)
sceneflow video.mp4 --save-frames --save-video --save-logs

# Save video to custom path (auto-enables --save-video)
sceneflow video.mp4 --output /path/to/my_cut_video.mp4

# Use LLM to select best frame
sceneflow video.mp4 --use-llm-selection --verbose

# Upload results to Airtable
sceneflow video.mp4 --airtable --verbose

# Disable energy refinement (use raw VAD)
sceneflow video.mp4 --no-energy-refinement

# Fine-tune energy refinement
sceneflow video.mp4 --energy-threshold-db 10.0 --energy-lookback-frames 25
```

### Python API

**Simple API (Recommended)**

```python
from sceneflow import get_cut_frame, get_ranked_cut_frames

# Get the single best cut point
best_time = get_cut_frame("video.mp4")
print(f"Cut at: {best_time:.2f}s")

# Get top 5 cut points
top_5 = get_ranked_cut_frames("video.mp4", n=5)
for i, time in enumerate(top_5, 1):
    print(f"{i}. {time:.2f}s")

# With custom configuration
from sceneflow import RankingConfig
config = RankingConfig(
    expression_neutrality_weight=0.40,  # Prioritize neutral expression
    motion_stability_weight=0.30
)
best_time = get_cut_frame("video.mp4", config=config, sample_rate=1)

# With LLM-powered frame selection (requires OPENAI_API_KEY)
best_time = get_cut_frame(
    "video.mp4",
    use_llm_selection=True,  # Uses GPT-4o to pick best from top 5
    openai_api_key="your-api-key"  # Or set OPENAI_API_KEY env var
)

# Upload results to Airtable
best_time = get_cut_frame(
    "video.mp4",
    upload_to_airtable=True,
    save_video=True  # Include cut video in Airtable
)

# Disable energy refinement (use raw VAD timestamp)
best_time = get_cut_frame(
    "video.mp4",
    use_energy_refinement=False
)

# Fine-tune energy refinement parameters
best_time = get_cut_frame(
    "video.mp4",
    energy_threshold_db=10.0,      # Require larger energy drop
    energy_lookback_frames=25      # Search further back from VAD
)
```

**Advanced API (More Control)**

```python
from sceneflow import CutPointRanker
from sceneflow.speech_detector import SpeechDetector
import cv2

# Detect when speech ends
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

# Get the best cut point
best_cut = ranked_frames[0]
print(f"Best cut point: {best_cut.timestamp:.2f}s (score: {best_cut.score:.4f})")
```

## How It Works

### 1. Speech Detection

SceneFlow uses Silero VAD, a deep learning model for voice activity detection. This provides highly accurate detection of when speech actually ends, not just when audio fades out.

### 1.5. Energy-Based Refinement

SceneFlow refines the VAD timestamp using audio energy analysis to find the exact frame where speech ends:
- Analyzes audio energy in a small window around the VAD timestamp
- Detects sharp energy drops (8+ dB) indicating true speech end
- Provides frame-accurate boundaries (typically adjusts VAD by 3-5 frames)

### 2. Visual Analysis

After identifying when speech ends, SceneFlow analyzes each frame using InsightFace (106-landmark facial detection) and ranks them based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Expression Neutrality** | 30% | Calm, neutral facial expressions (mouth closed, relaxed) |
| **Motion Stability** | 25% | Minimal optical flow between frames |
| **Eye Openness** | 20% | Natural eye openness (not too wide, not blinking) |
| **Pose Stability** | 15% | Steady head position and orientation |
| **Visual Sharpness** | 10% | Clear, sharp frame quality |

**Multi-Face Support**: When multiple faces are detected, SceneFlow uses center-weighted averaging to prioritize faces closer to the frame center while still considering all detected faces.

### 3. Ranking System

The algorithm uses a multi-stage ranking process:
1. Extract raw features from all frames
2. Normalize metrics across the entire frame set
3. Apply weighted scoring based on configuration
4. Use temporal context windows to favor stable sequences
5. Return ranked list of cut point candidates

## CLI Options

```bash
sceneflow SOURCE [OPTIONS]

Arguments:
  SOURCE                         Path to video file or URL

Output Options:
  --verbose                      Show detailed analysis information
  --json-output PATH             Save detailed analysis to JSON file (directory path)
  --top-n INT                    Return top N ranked timestamps in sorted order (shows scores)

Processing Options:
  --sample-rate INT              Process every Nth frame (default: 2)
  --save-frames                  Save annotated frames with InsightFace 106 landmarks
  --save-video                   Save cut video from start to best timestamp (requires ffmpeg)
  --output PATH                  Custom output path for saved video (auto-enables --save-video)
  --save-logs                    Save detailed feature extraction and scoring logs

Speech Detection Options:
  --no-energy-refinement         Disable energy-based refinement of VAD timestamp
  --energy-threshold-db FLOAT    Minimum dB drop for energy refinement (default: 8.0)
  --energy-lookback-frames INT   Max frames to search backward from VAD (default: 20)

Advanced Options:
  --use-llm-selection            Use GPT-4o vision to select best from top 5 (requires OPENAI_API_KEY)
  --airtable                     Upload results to Airtable (requires env vars)

  --help                         Show this message and exit
  --version                      Show version and exit
```

### Environment Variables

Set these environment variables to use optional features:

```bash
# Airtable Integration
AIRTABLE_ACCESS_TOKEN   Your Airtable access token
AIRTABLE_BASE_ID        Your Airtable base ID (e.g., appXXXXXXXXXXXXXX)
AIRTABLE_TABLE_NAME     Table name (optional, defaults to "SceneFlow Analysis")

# LLM-Powered Selection
OPENAI_API_KEY          OpenAI API key for GPT-4o vision integration
```

**Quick Setup:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual credentials
# The .env file is already in .gitignore for security
```

## API Reference

### `get_cut_frame()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str | *required* | Video file path or URL |
| `config` | RankingConfig | None | Custom scoring configuration |
| `sample_rate` | int | 2 | Process every Nth frame |
| `save_video` | bool | False | Save cut video (requires ffmpeg) |
| `save_frames` | bool | False | Save annotated frames |
| `save_logs` | bool | False | Save detailed analysis logs |
| `upload_to_airtable` | bool | False | Upload to Airtable |
| `use_llm_selection` | bool | False | Use GPT-4o for frame selection |
| `use_energy_refinement` | bool | True | Refine VAD with energy analysis |
| `energy_threshold_db` | float | 8.0 | Minimum dB drop for refinement |
| `energy_lookback_frames` | int | 20 | Max frames to search backward |
| `openai_api_key` | str | None | OpenAI API key (or env var) |
| `airtable_access_token` | str | None | Airtable token (or env var) |
| `airtable_base_id` | str | None | Airtable base ID (or env var) |
| `airtable_table_name` | str | None | Table name (or env var) |

### `RankingConfig` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expression_neutrality_weight` | float | 0.30 | Weight for neutral expressions |
| `motion_stability_weight` | float | 0.25 | Weight for low motion |
| `eye_openness_weight` | float | 0.20 | Weight for natural eye openness |
| `pose_stability_weight` | float | 0.15 | Weight for steady head position |
| `visual_sharpness_weight` | float | 0.10 | Weight for frame sharpness |
| `context_window_size` | int | 5 | Temporal smoothing window (odd) |
| `quality_gate_percentile` | float | 75.0 | Quality filtering threshold |
| `local_stability_window` | int | 5 | Stability analysis window (odd) |
| `center_weighting_strength` | float | 1.0 | Center bias for multi-face |
| `min_face_confidence` | float | 0.5 | Minimum face detection confidence |

**Note**: All weight parameters must sum to 1.0.

## Advanced Usage

### Custom Configuration

```python
from sceneflow import get_cut_frame, RankingConfig

# Emphasize neutral expression and motion stability
config = RankingConfig(
    expression_neutrality_weight=0.40,  # Prioritize closed mouth, neutral face
    motion_stability_weight=0.30,       # Minimal movement
    eye_openness_weight=0.15,
    pose_stability_weight=0.10,
    visual_sharpness_weight=0.05,
    context_window_size=7,              # Larger = more temporal smoothing
    quality_gate_percentile=80.0,       # Stricter quality filtering
    local_stability_window=7,           # Larger = favor longer stable sequences
    center_weighting_strength=2.0,      # Stronger center bias for multi-face
    min_face_confidence=0.7             # Higher confidence threshold
)

best_time = get_cut_frame("video.mp4", config=config)
```

### Save Outputs

```python
from sceneflow import get_cut_frame

# Save annotated frames and cut video
best_time = get_cut_frame(
    "video.mp4",
    save_frames=True,  # Saves frames with InsightFace 106 landmarks
    save_video=True,   # Saves cut video (requires ffmpeg)
    save_logs=True     # Saves detailed analysis logs
)

# Outputs saved to:
# - output/<video_name>/: Annotated frames (all detected faces)
# - output/<video_name>_cut.mp4: Cut video
# - logs/<video_name>_features.json: Feature extraction logs
# - logs/<video_name>_scores.json: Scoring logs
```

### Multiple Cut Points

```python
from sceneflow import get_ranked_cut_frames

# Get top 10 cut points
top_10 = get_ranked_cut_frames("video.mp4", n=10, sample_rate=2)

for i, timestamp in enumerate(top_10, 1):
    print(f"{i}. {timestamp:.2f}s")
```

## Example Output

### Default Mode
```bash
$ sceneflow video.mp4
5.23
```

### Verbose Mode
```bash
$ sceneflow video.mp4 --verbose

============================================================
SCENEFLOW - Smart Video Cut Point Detection
============================================================

Analyzing: video.mp4

[1/2] Detecting speech end time using VAD...
      Speech ends at: 8.45s (confidence: 0.95)
      Video duration: 10.50s

[2/2] Analyzing visual features from 8.45s to 10.50s...

============================================================
RESULTS
============================================================

Best cut point: 8.67s
Frame: 208
Score: 0.8745

Top 3 candidates:
  1. 8.67s (frame 208, score: 0.8745)
  2. 8.92s (frame 214, score: 0.8621)
  3. 9.15s (frame 220, score: 0.8534)

  ... and 47 more candidates
```

### With Energy Refinement (Default)
```bash
$ sceneflow video.mp4 --verbose

INFO: Stage 1: Detecting speech end time...
INFO: VAD detected speech end at: 8.45s
INFO: Stage 1.5: Refining speech end time with energy analysis...
INFO: Energy refinement adjusted timestamp by 3 frames
INFO: Analyzing frames from 8.32s to 10.50s
INFO: Best cut point found: 8.67s (score: 0.8745)
```

## Performance Tips

- **sample_rate**: Use `sample_rate=2` or `sample_rate=3` to skip frames for faster processing
- **Energy refinement**: Enabled by default, adds minimal overhead (~50-100ms)
- **Multi-face detection**: Performance scales with number of faces (~2-5ms per additional face)
- **LLM selection**: Adds ~1-2 seconds when enabled (requires OpenAI API call)
- **URL downloads**: Videos are automatically cleaned up after processing to save disk space

## Technical Details

- **Speech Detection**: Silero VAD (Voice Activity Detection) for accurate speech/silence detection
- **Energy Refinement**: Audio energy analysis with dB-based drop detection for frame-accurate boundaries
- **Facial Analysis**: InsightFace with 106-landmark detection (7-point eyes, 20-point mouth)
- **Multi-Face Support**: Center-weighted averaging prioritizes faces near frame center
- **Motion Analysis**: Farneback optical flow for motion stability
- **Eye Detection**: Eye Aspect Ratio (EAR) using 7-point eye landmarks
- **Expression Analysis**: Mouth Aspect Ratio (MAR) using 20-point mouth landmarks
- **Frame Quality**: Laplacian variance for sharpness assessment
- **Ranking**: Multi-factor scoring with temporal context windows
- **LLM Selection** *(Optional)*: GPT-4o vision analysis of top 5 candidates

## Use Cases

- **Content Creators**: Automatically find natural cut points in AI-generated talking head videos
- **Video Editors**: Quickly identify optimal endpoints for video clips
- **AI Video Tools**: Integrate smart cut point detection into video generation pipelines
- **Research**: Analyze facial features and speech patterns in videos

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details

## Examples

Check the `examples/` directory for more usage examples:
- `basic_usage.py` - Simple API usage
- `custom_config.py` - Custom configuration examples
- `ranked_results.py` - Getting multiple cut points
- `url_download.py` - Working with video URLs
- `batch_processing.py` - Processing multiple videos
- `save_logs_example.py` - Saving detailed analysis logs

For advanced features, see the documentation:
- `ENERGY_REFINEMENT.md` - Energy-based speech boundary refinement
- `MULTI_FACE_IMPLEMENTATION.md` - Multi-face detection and analysis
- `AIRTABLE_INTEGRATION.md` - Upload results to Airtable

## Links

- **Repository**: https://github.com/vertexcover-io/sceneflow
- **Issues**: https://github.com/vertexcover-io/sceneflow/issues
- **PyPI**: https://pypi.org/project/sceneflow/

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
