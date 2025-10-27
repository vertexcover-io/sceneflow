from sceneflow import CutPointRanker
import json
from pathlib import Path

video_path = "your_video.mp4"

ranker = CutPointRanker()

ranked_frames = ranker.rank_frames(
    video_path=video_path,
    start_time=5.0,
    end_time=10.0,
    sample_rate=2,
    save_frames=True,
    save_logs=True
)

print(f"Processed {len(ranked_frames)} frames")
print(f"Best cut point: {ranked_frames[0].timestamp:.2f}s")

video_base_name = Path(video_path).stem
log_dir = Path("output") / video_base_name

log_file = log_dir / f"rank_001_frame_{ranked_frames[0].frame_index}_timestamp_{ranked_frames[0].timestamp:.2f}.jsonl"

if log_file.exists():
    with open(log_file, 'r') as f:
        log_data = json.load(f)

    print("\nDetailed analysis for best frame:")
    print(f"  Eye openness (raw): {log_data['raw_features']['eye_openness']:.4f}")
    print(f"  Eye openness (score): {log_data['normalized_scores']['eye_openness_score']:.4f}")
    print(f"  Motion magnitude: {log_data['raw_features']['motion_magnitude']:.4f}")
    print(f"  Expression activity: {log_data['raw_features']['expression_activity']:.4f}")
    print(f"  Number of faces: {log_data['raw_features']['num_faces']}")

    if log_data['individual_faces']:
        print(f"\nFace details:")
        for face in log_data['individual_faces']:
            print(f"  Face {face['face_index']}: center_weight={face['center_weight']:.4f}")
