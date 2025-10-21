# Cut Point Ranker

A pure ranking algorithm for finding optimal cut points in AI-generated talking head videos.

## Features

- **No Thresholds**: Pure ranking approach - every frame is scored and ranked
- **Multi-Factor Analysis**: Combines 5 key metrics with configurable weights
- **Context-Aware**: Uses sliding window for temporal coherence
- **Research-Based**: Implements proven algorithms (EAR, optical flow, MediaPipe landmarks)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management (10-100x faster than pip).

### Install uv (if not already installed)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Project Dependencies

```bash
# Install all dependencies and create virtual environment
uv sync

# Or if you prefer the legacy method
pip install -r requirements.txt
```

## Quick Start

### Running Scripts with uv

```bash
# Run any Python script with uv
uv run python your_script.py

# Or activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Usage Example

```python
from cut_point_ranker import CutPointRanker

ranker = CutPointRanker()

ranked_frames = ranker.rank_frames(
    video_path="video.mp4",
    start_time=55.0,
    end_time=60.0,
    sample_rate=2
)

best_cut_point = ranked_frames[0]
print(f"Best cut at: {best_cut_point.timestamp:.2f}s")
```

## How It Works

### Ranking Factors

1. **Eye Openness (30%)** - Eye Aspect Ratio (EAR) detection
2. **Motion Stability (25%)** - Optical flow analysis (inverse)
3. **Expression Neutrality (20%)** - Facial expression activity (inverse)
4. **Pose Stability (15%)** - Head pose deviation (inverse)
5. **Visual Sharpness (10%)** - Laplacian variance

### Two-Pass Algorithm

**Pass 1: Feature Extraction**
- Extract all raw metrics from frames in the time range
- Calculate EAR, optical flow, blendshapes, pose, and sharpness

**Pass 2: Normalization & Ranking**
- Normalize all metrics to [0, 1] across the entire range
- Compute weighted composite scores
- Apply multi-frame context window (sliding average)
- Sort frames by final context score

### Multi-Frame Context

Uses a sliding window (default: 5 frames) to average composite scores. This ensures the selected cut point is part of a stable sequence, not just a single good frame.

## Custom Configuration

```python
from cut_point_ranker import CutPointRanker, RankingConfig

config = RankingConfig(
    eye_openness_weight=0.40,
    motion_stability_weight=0.30,
    expression_neutrality_weight=0.15,
    pose_stability_weight=0.10,
    visual_sharpness_weight=0.05,
    context_window_size=7
)

ranker = CutPointRanker(config)
```

## API Reference

### `CutPointRanker.rank_frames()`

Returns a ranked list of frames sorted by quality score.

**Parameters:**
- `video_path` (str): Path to video file
- `start_time` (float): Start timestamp in seconds
- `end_time` (float): End timestamp in seconds
- `sample_rate` (int): Process every Nth frame (default: 1)

**Returns:** `List[RankedFrame]`

### `CutPointRanker.get_detailed_scores()`

Returns detailed scoring breakdown for debugging.

**Returns:** `List[FrameScore]` with individual metric scores

## Project Structure

```
cut_point_ranker/
├── __init__.py         # Package exports
├── config.py           # Configuration dataclass
├── models.py           # Data models
├── extractors.py       # Feature extraction (MediaPipe, OpenCV)
├── normalizer.py       # Metric normalization utilities
├── scorer.py           # Scoring and context window logic
└── ranker.py           # Main pipeline orchestrator
```

## Algorithm Research Base

- **Eye Aspect Ratio**: Soukupová & Čech (2016)
- **Optical Flow**: Farneback method for motion estimation
- **Facial Landmarks**: MediaPipe FaceMesh (478 points + blendshapes)
- **Frame Quality**: Laplacian variance for sharpness assessment

## Example Output

```
Top 5 Cut Point Candidates:
Rank 1: Frame 8245 at 57.32s (score: 0.8745)
Rank 2: Frame 8312 at 57.79s (score: 0.8621)
Rank 3: Frame 8189 at 56.92s (score: 0.8534)
Rank 4: Frame 8401 at 58.42s (score: 0.8412)
Rank 5: Frame 8156 at 56.69s (score: 0.8301)
```

## Managing Dependencies with uv

### Adding New Dependencies

```bash
uv add package-name
uv add "package-name==1.2.3"  # Specific version
uv add "package-name>=1.0.0"  # Minimum version
```

### Removing Dependencies

```bash
uv remove package-name
```

### Updating Dependencies

```bash
uv lock --upgrade-package package-name  # Update specific package
uv lock --upgrade  # Update all packages
uv sync  # Sync environment with updated lockfile
```

### Why uv?

- **10-100x faster** than pip for package installation
- **All-in-one tool** - replaces pip, pip-tools, virtualenv, poetry, and more
- **Cross-platform lockfiles** (`uv.lock`) for reproducible builds
- **Built in Rust** for maximum performance
- **Python version management** built-in

## License

MIT
