"""
Test different EAR formulas to find the correct landmark mapping.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

LEFT_EYE_INDICES = list(range(35, 42))
RIGHT_EYE_INDICES = list(range(42, 49))


def test_ear_formulas(video_path: str, frame_idx: int = 100):
    """Test multiple EAR formula mappings to find the correct one."""

    # Open video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame")
        return

    # Initialize InsightFace
    app = FaceAnalysis(
        allowed_modules=['detection', 'landmark_2d_106'],
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_frame)

    if not faces or not hasattr(faces[0], 'landmark_2d_106'):
        print("No face or landmarks detected")
        return

    landmarks = faces[0].landmark_2d_106.astype(int)
    left_eye = landmarks[LEFT_EYE_INDICES]
    right_eye = landmarks[RIGHT_EYE_INDICES]

    print("="*60)
    print("LEFT EYE TOPOLOGY ANALYSIS")
    print("="*60)
    print("Coordinates (index: [x, y]):")
    for i, point in enumerate(left_eye):
        print(f"  {i}: [{point[0]:3d}, {point[1]:3d}]")

    # Identify key points by position
    print("\nAnalyzing positions...")

    # Find leftmost and rightmost (horizontal corners)
    leftmost_idx = np.argmin(left_eye[:, 0])
    rightmost_idx = np.argmax(left_eye[:, 0])
    print(f"Leftmost point: index {leftmost_idx}")
    print(f"Rightmost point: index {rightmost_idx}")

    # Find topmost and bottommost (vertical extremes)
    topmost_idx = np.argmin(left_eye[:, 1])
    bottommost_idx = np.argmax(left_eye[:, 1])
    print(f"Topmost point: index {topmost_idx}")
    print(f"Bottommost point: index {bottommost_idx}")

    print("\n" + "="*60)
    print("TESTING ALTERNATIVE EAR FORMULAS")
    print("="*60)

    def calc_ear(eye, h_indices, v1_indices, v2_indices):
        """
        Generic EAR calculator.
        h_indices: [left, right] for horizontal distance
        v1_indices: [top, bottom] for first vertical distance
        v2_indices: [top, bottom] for second vertical distance
        """
        A = np.linalg.norm(eye[v1_indices[0]] - eye[v1_indices[1]])
        B = np.linalg.norm(eye[v2_indices[0]] - eye[v2_indices[1]])
        C = np.linalg.norm(eye[h_indices[0]] - eye[h_indices[1]])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    # Test Formula 1: Current (wrong) formula
    ear1_left = calc_ear(left_eye, [0, 3], [1, 5], [2, 4])
    ear1_right = calc_ear(right_eye, [0, 3], [1, 5], [2, 4])
    print(f"\n1. CURRENT FORMULA [H: 0-3, V1: 1-5, V2: 2-4]:")
    print(f"   Left: {ear1_left:.4f}, Right: {ear1_right:.4f}, Avg: {(ear1_left + ear1_right)/2:.4f}")

    # Test Formula 2: Leftmost-Rightmost horizontal, with vertical pairs
    ear2_left = calc_ear(left_eye, [0, 4], [5, 1], [5, 2])
    ear2_right = calc_ear(right_eye, [1, 4], [6, 2], [6, 3])  # Right eye might be mirrored
    print(f"\n2. FORMULA [H: 0-4, V1: 5-1, V2: 5-2]:")
    print(f"   Left: {ear2_left:.4f}, Right: {ear2_right:.4f}, Avg: {(ear2_left + ear2_right)/2:.4f}")

    # Test Formula 3: Using detected extremes
    ear3_left = calc_ear(left_eye, [leftmost_idx, rightmost_idx],
                        [topmost_idx, bottommost_idx],
                        [topmost_idx, bottommost_idx])
    # For right eye, detect separately
    right_leftmost_idx = np.argmin(right_eye[:, 0])
    right_rightmost_idx = np.argmax(right_eye[:, 0])
    right_topmost_idx = np.argmin(right_eye[:, 1])
    right_bottommost_idx = np.argmax(right_eye[:, 1])
    ear3_right = calc_ear(right_eye, [right_leftmost_idx, right_rightmost_idx],
                         [right_topmost_idx, right_bottommost_idx],
                         [right_topmost_idx, right_bottommost_idx])
    print(f"\n3. EXTREME-BASED FORMULA [H: extreme-extreme, V: extreme-extreme]:")
    print(f"   Left: {ear3_left:.4f}, Right: {ear3_right:.4f}, Avg: {(ear3_left + ear3_right)/2:.4f}")

    # Test Formula 4: Standard 6-point EAR (MediaPipe-style)
    # Assuming points are ordered: corner1, top1, top2, corner2, bottom2, bottom1
    ear4_left = calc_ear(left_eye, [0, 4], [5, 1], [3, 2])
    ear4_right = calc_ear(right_eye, [1, 4], [6, 2], [5, 3])
    print(f"\n4. STANDARD 6-POINT [H: 0-4, V1: 5-1, V2: 3-2]:")
    print(f"   Left: {ear4_left:.4f}, Right: {ear4_right:.4f}, Avg: {(ear4_left + ear4_right)/2:.4f}")

    # Test Formula 5: Simplified with center points
    ear5_left = calc_ear(left_eye, [0, 4], [5, 2], [6, 1])
    ear5_right = calc_ear(right_eye, [1, 4], [6, 3], [5, 2])
    print(f"\n5. CENTER-BASED [H: 0-4, V1: 5-2, V2: 6-1]:")
    print(f"   Left: {ear5_left:.4f}, Right: {ear5_right:.4f}, Avg: {(ear5_left + ear5_right)/2:.4f}")

    print("\n" + "="*60)
    print("EXPECTED RANGE: 0.25 - 0.35 for normal open eyes")
    print("="*60)
    print("\nRecommendation: Choose the formula with avg closest to 0.30")


if __name__ == "__main__":
    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/002_explainer.mp4"
    test_ear_formulas(video_path, frame_idx=100)
