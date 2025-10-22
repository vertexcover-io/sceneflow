from cut_point_ranker import CutPointRanker, RankingConfig
from speech_detector import SpeechTimestampDetector
import os
from pathlib import Path
import cv2
import json


def speech_detection_comparison():
    """Compare all speech detection methods"""
    detector = SpeechTimestampDetector(model_size="small")

    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/New/aJQBKYmCN8XuO6MF.mp4"

    print("\n=== METHOD 1: Standard Whisper Detection ===")
    speech_end_time = detector.get_speech_end_time(video_path, save_transcription=True)
    print(f"Speech ends at: {speech_end_time:.2f} seconds")
    print("Method: Whisper transcription only")

    print("\n=== METHOD 2: Precise Energy-Gradient Detection ===")
    precise_end_time, confidence = detector.get_precise_speech_end_time(
        video_path,
        return_confidence=True,
        save_transcription=False
    )
    print(f"Precise speech end: {precise_end_time:.2f} seconds (confidence: {confidence:.2f})")
    print(f"Difference from Whisper: {abs(precise_end_time - speech_end_time):.3f} seconds")
    print("Method: Whisper + energy gradient analysis")

    print("\n=== METHOD 3: Silero VAD Detection ===")
    try:
        vad_end_time, vad_confidence = detector.get_vad_speech_end_time(
            video_path,
            return_confidence=True
        )
        print(f"VAD speech end: {vad_end_time:.2f} seconds (confidence: {vad_confidence:.2f})")
        print(f"Difference from Whisper: {abs(vad_end_time - speech_end_time):.3f} seconds")
        print("Method: Deep learning VAD model (Silero)")
    except RuntimeError as e:
        print(f"VAD detection unavailable: {e}")

    print("\n=== METHOD 4: Hybrid Multi-Method Detection (RECOMMENDED) ===")
    try:
        hybrid_end_time, details = detector.get_hybrid_speech_end_time(
            video_path,
            return_details=True
        )
        print(f"Hybrid speech end: {hybrid_end_time:.2f} seconds")
        print(f"Final confidence: {details['final_confidence']:.2f}")
        print("\nDetailed breakdown:")
        print(f"  - Whisper estimate: {details['whisper_end']:.2f}s")
        print(f"  - VAD refinement: {details['vad_end']:.2f}s")
        print(f"  - Energy refinement: {details['energy_end']:.2f}s")
        print(f"  - Energy confidence: {details['energy_confidence']:.2f}")
        print(f"  - Spectral confidence: {details['spectral_confidence']:.2f}")
        print(f"  - Agreement score: {details['agreement_score']:.2f}")
        print("Method: Whisper + Silero VAD + Energy Analysis + Spectral Features")
    except RuntimeError as e:
        print(f"Hybrid detection unavailable: {e}")

    print("\n=== COMPARISON SUMMARY ===")
    print(f"Standard Whisper:     {speech_end_time:.2f}s")
    print(f"Precise Energy:       {precise_end_time:.2f}s (Δ {abs(precise_end_time - speech_end_time):.3f}s)")
    try:
        print(f"Silero VAD:           {vad_end_time:.2f}s (Δ {abs(vad_end_time - speech_end_time):.3f}s)")
        print(f"Hybrid (Recommended): {hybrid_end_time:.2f}s (Δ {abs(hybrid_end_time - speech_end_time):.3f}s)")
    except:
        pass


def speech_detection_example():
    """Detect when speech ends in a talking head video (using hybrid method)"""
    detector = SpeechTimestampDetector(model_size="small")

    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/video_16_9.mp4"

    # Use the recommended hybrid detection method
    print("\n=== Hybrid Speech Detection (Recommended) ===")
    try:
        speech_end_time, confidence = detector.get_hybrid_speech_end_time(
            video_path,
            return_confidence=True,
            save_transcription=True
        )
        print(f"Speech ends at: {speech_end_time:.2f} seconds (confidence: {confidence:.2f})")
    except RuntimeError:
        # Fallback to precise method if VAD not available
        print("Note: VAD not available, using precise energy-gradient method")
        speech_end_time, confidence = detector.get_precise_speech_end_time(
            video_path,
            return_confidence=True,
            save_transcription=True
        )
        print(f"Speech ends at: {speech_end_time:.2f} seconds (confidence: {confidence:.2f})")

    if speech_end_time:
        print(f"\nStarting visual analysis from: {speech_end_time:.2f}s")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        ranker = CutPointRanker()
        ranked_frames = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,  # Use hybrid detection result
            end_time=duration,
            sample_rate=2,
            save_video=True
        )

        if ranked_frames:
            best_frame = ranked_frames[0]
            print(f"Best cut point: Frame {best_frame.frame_index} at {best_frame.timestamp:.2f}s (score: {best_frame.score:.4f})")
    else:
        print("No speech detected in the video")


def basic_usage():
    """Basic usage with VAD speech detection"""
    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/New/8g9Rq5Hgnh8wNRuq.mp4"

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    # Use VAD detection to get speech end time
    print("Detecting speech end time using VAD...")
    detector = SpeechTimestampDetector(model_size="small")

    try:
        start_time, confidence = detector.get_vad_speech_end_time(
            video_path,
            return_confidence=True,
        )
        print(f"VAD speech end: {start_time:.2f}s (confidence: {confidence:.2f})")
    except RuntimeError as e:
        # Fallback to precise method if VAD not available
        print(f"VAD not available: {e}")
        print("Falling back to precise energy-gradient detection...")
        start_time, confidence = detector.get_precise_speech_end_time(
            video_path,
            return_confidence=True,
            save_transcription=True
        )
        print(f"Precise speech end: {start_time:.2f}s (confidence: {confidence:.2f})")

    end_time = duration

    # Rank frames after speech ends
    print(f"\nAnalyzing frames from {start_time:.2f}s to {end_time:.2f}s")
    ranker = CutPointRanker()
    ranked_frames = ranker.rank_frames(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        sample_rate=2,
        save_video=True
    )

    print("\nTop 5 Cut Point Candidates:")
    for frame in ranked_frames[:5]:
        print(f"Rank {frame.rank}: Frame {frame.frame_index} at {frame.timestamp:.2f}s (score: {frame.score:.4f})")


def detailed_analysis():
    """Get detailed scoring breakdown for debugging"""
    ranker = CutPointRanker()

    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/011_explainer.mp4"
    start_time = 5.292
    end_time = 8.0

    detailed_scores = ranker.get_detailed_scores(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        sample_rate=1
    )

    print("\nDetailed Score Breakdown (Top 3):")
    for i, score in enumerate(detailed_scores[:3], 1):
        print(f"\n=== Rank {i}: Frame {score.frame_index} at {score.timestamp:.2f}s ===")
        print(f"Final Score: {score.final_score:.4f}")
        print(f"Composite Score: {score.composite_score:.4f}")
        print(f"Quality Penalty: {score.quality_penalty:.4f}")
        print(f"Stability Boost: {score.stability_boost:.4f}")
        print(f"\nComponent Scores:")
        print(f"  - Eye Openness: {score.eye_openness_score:.4f}")
        print(f"  - Motion Stability: {score.motion_stability_score:.4f}")
        print(f"  - Expression Neutrality: {score.expression_neutrality_score:.4f}")
        print(f"  - Pose Stability: {score.pose_stability_score:.4f}")
        print(f"  - Visual Sharpness: {score.visual_sharpness_score:.4f}")


def custom_weights():
    """Use custom weighting and multi-stage parameters"""
    config = RankingConfig(
        eye_openness_weight=0.40,
        motion_stability_weight=0.30,
        expression_neutrality_weight=0.15,
        pose_stability_weight=0.10,
        visual_sharpness_weight=0.05,
        context_window_size=7,
        quality_gate_percentile=80.0,
        local_stability_window=7
    )

    ranker = CutPointRanker(config)

    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/003_explainer.mp4"
    start_time = 5.151
    end_time = 8.0

    ranked_frames = ranker.rank_frames(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time
    )

    best_frame = ranked_frames[0]
    print(f"\nBest cut point: Frame {best_frame.frame_index} at {best_frame.timestamp:.2f}s")


def batch_usage(video_dir, start_time=None, end_time=None, sample_rate=2, use_speech_detection=True, detection_mode='hybrid', save_transcriptions=True):
    """Process all videos in a directory

    Args:
        video_dir: Directory containing video files
        start_time: Start time for analysis (default: None, will use speech detection if enabled)
        end_time: End time for analysis (default: None, will use video duration if needed)
        sample_rate: Frame sampling rate (default: 2)
        use_speech_detection: Whether to use speech detection to find start time (default: True)
        detection_mode: Speech detection method: 'standard', 'precise', 'vad', or 'hybrid' (default: 'hybrid')
        save_transcriptions: Whether to save transcription data to JSON files (default: True)
    """
    ranker = CutPointRanker()

    # Initialize speech detector if needed
    detector = None
    if use_speech_detection:
        print("Initializing speech detector...")
        detector = SpeechTimestampDetector(model_size="base")
        print("Speech detector ready\n")

    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

    # Get all video files from directory
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"Error: Directory '{video_dir}' does not exist")
        return

    video_files = [f for f in video_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"No video files found in '{video_dir}'")
        return

    print(f"Found {len(video_files)} video(s) to process\n")

    results = {}

    for i, video_path in enumerate(sorted(video_files), 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 60)

        try:
            video_start_time = start_time
            video_end_time = end_time

            # Get video duration
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()

            # Use speech detection if enabled and no start_time provided
            if use_speech_detection and video_start_time is None and detector is not None:
                print(f"  Detecting speech end time (mode: {detection_mode})...")

                try:
                    speech_end_time, confidence = detector.get_vad_speech_end_time(
                            str(video_path),
                            return_confidence=True,
                            save_transcription=save_transcriptions
                        )

                    if speech_end_time:
                        video_start_time = speech_end_time
                        print(f"  Speech end: {speech_end_time:.2f}s (confidence: {confidence:.2f})")
                        print(f"  Starting visual analysis from: {speech_end_time:.2f}s")
                    else:
                        print(f"  No speech detected, using default start time (5.0s)")
                        video_start_time = 5.0
                except RuntimeError as e:
                    # Fallback if VAD not available
                    print(f"  {e}")
                    print(f"  Falling back to precise energy-gradient detection")
                    speech_end_time, confidence = detector.get_precise_speech_end_time(
                        str(video_path),
                        return_confidence=True,
                        save_transcription=save_transcriptions
                    )
                    if speech_end_time:
                        video_start_time = speech_end_time
                        print(f"  Speech end: {speech_end_time:.2f}s (confidence: {confidence:.2f})")
                    else:
                        print(f"  No speech detected, using default start time (5.0s)")
                        video_start_time = 5.0

            elif video_start_time is None:
                video_start_time = 5.0

            # Set end time to video duration if not specified
            if video_end_time is None:
                video_end_time = duration

            ranked_frames = ranker.rank_frames(
                video_path=str(video_path),
                start_time=video_start_time,
                end_time=video_end_time,
                sample_rate=sample_rate,
                save_video=True
            )

            if ranked_frames:
                best_frame = ranked_frames[0]
                print(f"  Best cut point: Frame {best_frame.frame_index} at {best_frame.timestamp:.2f}s (score: {best_frame.score:.4f})")
                print(f"  Top 3 candidates:")
                for frame in ranked_frames[:3]:
                    print(f"    Rank {frame.rank}: Frame {frame.frame_index} at {frame.timestamp:.2f}s (score: {frame.score:.4f})")

                results[video_path.name] = {
                    'best_frame': best_frame.frame_index,
                    'best_timestamp': best_frame.timestamp,
                    'best_score': best_frame.score,
                    'top_frames': ranked_frames[:5],
                    'speech_end_time': video_start_time if use_speech_detection else None,
                    'transcription_file': video_path.stem + '_transcription.json' if save_transcriptions and use_speech_detection else None
                }
            else:
                print(f"  No frames found in the specified time range")
                results[video_path.name] = None

        except Exception as e:
            print(f"  Error processing video: {str(e)}")
            results[video_path.name] = None

        print()

    # Summary
    print("=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    successful = sum(1 for v in results.values() if v is not None)
    print(f"Successfully processed: {successful}/{len(video_files)} videos")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CUT POINT RANKER - EXAMPLE USAGE")
    print("=" * 60)

    # Uncomment the example you want to run:

    # Compare all speech detection methods
    # print("\n1. Speech Detection Methods Comparison")
    # print("-" * 60)
    # speech_detection_comparison()

    # Standard workflow with hybrid detection (recommended)
    # print("\n2. Speech Detection + Visual Analysis (Hybrid)")
    # print("-" * 60)
    # speech_detection_example()

    # Basic usage without speech detection
    print("\n3. Basic Usage")
    print("-" * 60)
    basic_usage()

    # Get detailed scoring breakdown
    # print("\n4. Detailed Analysis")
    # print("-" * 60)
    # detailed_analysis()

    # Use custom weighting configuration
    # print("\n5. Custom Weights")
    # print("-" * 60)
    # custom_weights()

    # Process multiple videos at once
    # print("\n6. Batch Processing")
    # print("-" * 60)
    # batch_usage(
    #     video_dir="D:/vertexcover/ai-video-cutter/dataset/AI/New",
    #     sample_rate=2,
    #     detection_mode='hybrid'  # Use 'standard', 'precise', 'vad', or 'hybrid'
    # )
