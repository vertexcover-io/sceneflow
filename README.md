# SceneFlow

**Smart cut point detection for AI-generated talking head videos**

SceneFlow automatically finds the optimal point to cut your AI-generated talking head videos using advanced speech detection and multi-factor visual analysis. No more awkward mid-speech cuts or unwanted motion at the end of your videos!

## Features

- **Intelligent Speech Detection** - Uses Silero VAD (Voice Activity Detection) to precisely identify when speech ends
- **Multi-Factor Visual Analysis** - Ranks potential cut points based on:
  - Eye openness (natural blink detection)
  - Motion stability (minimal movement)
  - Expression neutrality (calm facial expressions)
  - Pose stability (steady head position)
  - Visual sharpness (frame quality)
- **URL Support** - Works directly with video URLs (YouTube, etc.) via yt-dlp
- **Flexible Output** - Simple timestamp output, verbose mode, or detailed JSON analysis
- **Customizable Weights** - Adjust the importance of each visual factor
- **Pure Ranking Algorithm** - No arbitrary thresholds, everything is relative

## Installation

```bash
pip install sceneflow
```

### Requirements

- Python 3.9 or higher
- FFmpeg (for video processing)

## Quick Start

### Command Line

```bash
# Basic usage - outputs just the timestamp
sceneflow video.mp4
# Output: 5.23

# Verbose mode - see detailed analysis
sceneflow video.mp4 --verbose

# Save detailed JSON analysis
sceneflow video.mp4 --json ./output

# From a URL
sceneflow "https://www.youtube.com/watch?v=..." --verbose

# Custom weights
sceneflow video.mp4 --eye-weight 0.4 --motion-weight 0.3
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
    eye_openness_weight=0.40,
    motion_stability_weight=0.30
)
best_time = get_cut_frame("video.mp4", config=config, sample_rate=1)
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

### 2. Visual Analysis

After identifying when speech ends, SceneFlow analyzes each frame using MediaPipe FaceMesh (478 facial landmarks + 52 blendshapes) and ranks them based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Eye Openness** | 30% | Prefers natural eye openness (not too wide, not blinking) |
| **Motion Stability** | 25% | Minimal optical flow between frames |
| **Expression Neutrality** | 20% | Calm, neutral facial expressions |
| **Pose Stability** | 15% | Steady head position and orientation |
| **Visual Sharpness** | 10% | Clear, sharp frame quality |

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
  SOURCE                  Path to video file or URL

Options:
  --verbose              Show detailed analysis information
  --json-output PATH     Save detailed analysis to JSON file (directory path)
  --sample-rate INT      Process every Nth frame (default: 2)
  --save-frames          Save annotated frames with MediaPipe landmarks
  --save-video           Save cut video from start to best timestamp (requires ffmpeg)
  --top-n INT            Return top N ranked timestamps in sorted order (shows scores)

  --help                 Show this message and exit
  --version              Show version and exit
```

## Advanced Usage

### Custom Configuration

```python
from sceneflow import get_cut_frame, RankingConfig

# Emphasize eye openness and motion stability
config = RankingConfig(
    eye_openness_weight=0.40,
    motion_stability_weight=0.30,
    expression_neutrality_weight=0.15,
    pose_stability_weight=0.10,
    visual_sharpness_weight=0.05,
    context_window_size=7,          # Larger = more temporal smoothing
    quality_gate_percentile=80.0,   # Stricter quality filtering
    local_stability_window=7        # Larger = favor longer stable sequences
)

best_time = get_cut_frame("video.mp4", config=config)
```

### Save Outputs

```python
from sceneflow import get_cut_frame

# Save annotated frames and cut video
best_time = get_cut_frame(
    "video.mp4",
    save_frames=True,  # Saves frames with MediaPipe landmarks
    save_video=True    # Saves cut video (requires ffmpeg)
)

# Outputs saved to:
# - output/<video_name>/: Annotated frames
# - output/<video_name>_cut.mp4: Cut video
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

[1/2] Detecting speech end time (model: small)...
      Speech ends at: 5.12s (confidence: 0.87)
      Video duration: 8.50s

[2/2] Analyzing visual features from 5.12s to 8.50s...

============================================================
RESULTS
============================================================

Best cut point: 5.23s
Frame: 157
Score: 0.8745

Top 3 candidates:
  1. 5.23s (frame 157, score: 0.8745)
  2. 5.45s (frame 164, score: 0.8621)
  3. 5.89s (frame 177, score: 0.8534)

  ... and 47 more candidates
```

## Performance Tips

- **sample_rate**: Use `sample_rate=2` or `sample_rate=3` to skip frames for faster processing
- **Whisper model**: Default `small` model provides good balance of speed and accuracy
- **URL downloads**: Videos are automatically cleaned up after processing to save disk space

## Technical Details

- **Speech Detection**: Silero VAD (Voice Activity Detection) for accurate speech/silence detection
- **Facial Analysis**: MediaPipe FaceMesh (478 landmarks + 52 blendshapes)
- **Motion Analysis**: Farneback optical flow for motion stability
- **Eye Detection**: Eye Aspect Ratio (EAR) method for blink detection
- **Frame Quality**: Laplacian variance for sharpness assessment
- **Ranking**: Multi-factor scoring with temporal context windows

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

## Links

- **Repository**: https://github.com/vertexcover-io/sceneflow
- **Issues**: https://github.com/vertexcover-io/sceneflow/issues
- **PyPI**: https://pypi.org/project/sceneflow/

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
